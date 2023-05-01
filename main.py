import os
from flask import Flask, request, jsonify, send_from_directory
import zipfile
import json
import glob
import subprocess
import threading
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
def start_training():
    try:
        # Start the training process in a separate thread
        thread = threading.Thread(target=train_and_output, args=(25, 4))
        thread.start()
        return jsonify({"message": "Training started successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error starting training: {e}"}), 500


def train_and_output(EPOCHS=25, BATCH_SIZE=4):
    if EPOCHS is None:
        EPOCHS = 25
    if BATCH_SIZE is None:
        BATCH_SIZE = 4
    print("Starting training")
    freeze_list = [int(x) for x in "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14".split()]
    command = [
        "python",
        yolov5_dir + "/train.py",
        "--data",
        "./data.yaml",
        "--weights",
        f"{yolov5_dir}/models/yolov5m.pt",
        "--img",
        "640",
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--name",
        RES_DIR,
        "--device",
        "0",
        "--freeze",
    ]
    command.extend(map(str, freeze_list))
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Print the output of the subprocess in the console
    for line in process.stdout:
        print(line.strip())

    # Wait for training to finish
    process.communicate()

    # Return response immediately
    message = f"Training started with {EPOCHS} epochs and {BATCH_SIZE} batch size in {RES_DIR}."
    return jsonify({"message": message}), 200


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
