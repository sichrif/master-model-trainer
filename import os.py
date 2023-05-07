import os
from flask import Flask, request, jsonify, send_from_directory
import zipfile
import json
import glob
import subprocess
import websockets
import asyncio
from datetime import datetime
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    JWTManager,
)


app = Flask(__name__)

# configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://root:root@localhost/pfe"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

db = SQLAlchemy(app)
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)

# initialize bcrypt
bcrypt = Bcrypt(app)

jwt = JWTManager(app)


# create User model
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now())

    def __init__(self, username, password):
        self.username = username
        self.password = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)


# create Results model
class Results(db.Model):
    __tablename__ = "model_results"

    id = db.Column(db.Integer, primary_key=True)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now())


yolov5_dir = os.path.join(os.getcwd(), "yolov5")


def set_res_dir():
    if not os.getcwd() == yolov5_dir:
        print("changing directory to yolov5 2")
        subprocess.run(["cd", "yolov5"], shell=True)
    # Directory to store results
    res_dir_count = len(glob.glob(yolov5_dir + "/runs/train/*"))
    print(f"Current number of result directories: {res_dir_count}")
    RES_DIR = f"results_{res_dir_count}"
    return RES_DIR


async def train_and_output(websocket):
    print("result directory: ")
    # os.chdir("yolov5")
    EPOCHS = 25
    BATCH_SIZE = 4
    # if EPOCHS is None:
    #     EPOCHS = 25
    # if BATCH_SIZE is None:
    #     BATCH_SIZE = 4
    print("Sending training updates")
    command = [
        "python",
        "train.py",
        "--data",
        "../data.yaml",
        "--weights",
        "yolov5s.pt",
        "--img",
        "640",
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        "4",
        "--name",
        RES_DIR,
        "--device",
        "0",
    ]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print("Training started", process)
    # Wait for training to finish
    while True:
        try:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            message = {"type": "output", "data": output.strip()}
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosedError:
            print("WebSocket connection closed unexpectedly")
            break

    # Calculate evaluation metrics
    eval_command = [
        "python",
        "test.py",
        "--weights",
        f"runs/train/{RES_DIR}/weights/best.pt",
        "--data",
        "../data.yaml",
        "--img-size",
        "640",
        "--conf-thres",
        "0.001",
        "--iou-thres",
        "0.6",
    ]

    eval_process = subprocess.Popen(
        eval_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Extract the evaluation metrics
    precision = float(eval_process.stdout.split("precision = ")[1].split("\n")[0])
    recall = float(eval_process.stdout.split("recall = ")[1].split("\n")[0])
    f1_score = float(eval_process.stdout.split("F1-score = ")[1].split("\n")[0])

    # Save the results to the database
    result = Results(RES_DIR, precision, recall, f1_score)
    db.session.add(result)
    db.session.commit()

    # Send message to WebSocket when training is done
    message = f"Training is done! {EPOCHS} epochs were completed in {RES_DIR}."
    await websocket.send(message)


async def main():
    async with websockets.serve(train_and_output, "localhost", 8765, ping_timeout=None):
        await asyncio.Future()  # run forever


RES_DIR = set_res_dir()


@app.route("/process", methods=["POST"])
def process_zip():
    # Get the uploaded file
    file = request.files["file"]
    file.save("data.zip")

    # Delete the directories if found
    dirs = ["train", "valid", "test"]
    for dir_name in dirs:
        if os.path.exists(dir_name):
            os.system(f"rmdir /s /q {dir_name}")
    # Extract the uploaded file
    with zipfile.ZipFile(file) as zip_file:
        zip_file.extractall()
    # Delete the uploaded file
    os.remove("data.zip")
    return "Processing complete!"


@app.route("/detect", methods=["GET"])
@jwt_required()
def start_detection():
    if os.getcwd() == yolov5_dir:
        print("changing directory to yolov5 1")
        os.chdir(yolov5_dir)
    # Directory to store inference results.
    infer_dir_count = len(glob.glob(yolov5_dir + "/runs/detect/*"))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    # Inference on video stream.
    subprocess.Popen(
        [
            "python",
            yolov5_dir + "/detect.py",
            "--weights",
            f"{yolov5_dir}/runs/train/{RES_DIR}/weights/best.pt",
            "--source",
            "0",
            "--name",
            INFER_DIR,
            "--device",
            "0",
            "--save-txt",
            "--conf",
            "0.4",
        ]
    )
    mp4_url = f"http://localhost:5000/{INFER_DIR}/0.mp4"
    return (
        jsonify({"message": "Detection job started successfully", "mp4_url": mp4_url}),
        200,
    )


@app.route("/train", methods=["GET"])
def start_websocket():
    try:
        asyncio.run(main())
        return jsonify({"message": "WebSocket started successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error starting WebSocket: {e}"}), 500


# @app.route("/reports/<path:path>")
# def send_report(path):
#     infer_dir_count = len(glob.glob(yolov5_dir + "/runs/detect/*"))
#     return send_from_directory(
#         yolov5_dir + "/runs/detect/" + "ineference_" + infer_dir_count + "0.mp4" , path
#     )


# register endpoint
@app.route("/register", methods=["POST"])
def register():
    username = request.json.get("username")
    password = request.json.get("password")

    if not username or not password:
        return jsonify({"msg": "Username and password are required."}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists."}), 400

    user = User(username, password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"msg": "Registration successful."}), 201


# login endpoint
@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({"msg": "Invalid username or password."}), 401

    access_token = create_access_token(identity=user.id)
    return jsonify({"access_token": access_token}), 200


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    return response


if __name__ == "__main__":
    app.run(port=5000, debug=True)
