import os
import shutil
import signal
import uuid
from flask import Flask, request, jsonify, send_from_directory, url_for
import zipfile
import glob
import subprocess
import threading
from datetime import datetime
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    jwt_required,
    JWTManager,
)


app = Flask(__name__)

# configure database
app.config[
    "SQLALCHEMY_DATABASE_URI"
] = "postgresql://yolo-user:yolo-password@localhost/yolo-db"
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

    user_id = db.Column(db.String(80), primary_key=True)
    firstname = db.Column(db.String(80), nullable=True)
    lastname = db.Column(db.String(80), nullable=True)
    phonenum = db.Column(db.String(80), nullable=True)
    email = db.Column(db.String(80), nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now())

    def __init__(self, username, password, firstname, lastname, phonenum, email):
        self.user_id = str(uuid.uuid4())
        self.username = username
        self.password = bcrypt.generate_password_hash(password).decode("utf-8")
        self.firstname = firstname
        self.lastname = lastname
        self.phonenum = phonenum
        self.email = email

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)


# create Model class
class Model(db.Model):
    __tablename__ = "models"

    model_id = db.Column(db.String(80), primary_key=True)
    model_name = db.Column(db.String(80), nullable=False)
    model_image = db.Column(db.String(150), nullable=True)
    user_id = db.Column(db.String(80), db.ForeignKey("users.user_id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now())

    def __init__(self, model_name, model_image, user_id):
        self.model_id = str(uuid.uuid4())
        self.model_name = model_name
        self.model_image = model_image
        self.user_id = user_id


# create required tables if they don't exist
with app.app_context():
    db.create_all()
yolov5_dir = os.path.join(os.getcwd(), "yolov5")


def set_res_dir(user_id):
    if not os.getcwd() == yolov5_dir:
        print("changing directory to yolov5 2")
        subprocess.run(["cd", "yolov5"], shell=True)
    # Directory to store results
    res_dir_count = len(glob.glob(yolov5_dir + f"/runs/train/{user_id}/*"))
    print(f"Current number of result directories for user {user_id}: {res_dir_count}")
    RES_DIR = f"results_{res_dir_count}"
    return RES_DIR


@app.route("/process", methods=["POST"])
@jwt_required()
def process_zip():
    # Add the new model to the database
    user_id = get_jwt_identity()
    model_name = request.form.get("model_name")
    model_image = request.form.get("model_image")
    # Get the uploaded file
    file = request.files["file"]
    if not model_name or not file:
        return jsonify({"message": "File and model name are required."}), 400

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
    model = Model(model_name, model_image, user_id)
    db.session.add(model)
    db.session.commit()

    return "Processing complete!"


@app.route("/detect", methods=["GET"])
@jwt_required()
def start_detection():
    user_id = get_jwt_identity()
    model_name = request.args.get("model_name")
    if os.getcwd() == yolov5_dir:
        print("changing directory to yolov5 1")
        os.chdir(yolov5_dir)
    # Directory to store inference results.
    infer_dir_count = len(
        glob.glob(yolov5_dir + f"/runs/detect/{user_id}_{model_name}/*")
    )
    # Directory to store inference results.
    results_dir_count = len(
        glob.glob(yolov5_dir + f"/runs/train/{user_id}/{model_name}/*")
    )
    print(
        f"Current number of inference detection directories for user {user_id} and model {model_name}: {infer_dir_count}"
    )
    INFER_DIR = f"inference_{infer_dir_count+1}"
    TRAIN_DIR = f"results_{results_dir_count+1}"
    # Inference on video stream.
    process = subprocess.Popen(
        [
            "python",
            yolov5_dir + "/detect.py",
            "--weights",
            f"{yolov5_dir}/runs/train/{user_id}/{model_name}/{TRAIN_DIR}/weights/best.pt",
            "--source",
            "0",
            "--name",
            f"{user_id}_{model_name}_{INFER_DIR}",
            "--device",
            "0",
            "--save-txt",
            "--conf",
            "0.4",
        ]
    )
    process_id = process.pid
    mp4_url = f"http://localhost:5000/{user_id}_{model_name}_{INFER_DIR}/0.mp4"
    return (
        jsonify(
            {
                "message": "Detection job started successfully",
                "mp4_url": mp4_url,
                "process_id": process_id,
            }
        ),
        200,
    )


@app.route("/get_training_details", methods=["GET"])
@jwt_required()
def get_training_details():
    user_id = get_jwt_identity()
    model_id = request.args.get("model_id")
    if not model_id:
        return (
            jsonify({"message": "Model ID is required."}),
            400,
        )

    if os.getcwd() == yolov5_dir:
        print("changing directory to yolov5 1")
        os.chdir(yolov5_dir)

    # Get the model name from the model ID
    model = Model.query.filter_by(user_id=user_id, model_id=model_id).first()
    if not model:
        return jsonify({"message": "Model not found."}), 404

    model_name = model.model_name

    # Directory to store inference results.
    results_dir_count = len(
        glob.glob(yolov5_dir + f"/runs/train/{user_id}/{model_name}/*")
    )
    TRAIN_DIR = f"results_{results_dir_count+1}"

    training_dir = os.path.join(
        yolov5_dir, "runs", "train", user_id, model_name, TRAIN_DIR
    )
    print(training_dir)
    if not os.path.exists(training_dir):
        return jsonify({"message": "Training directory not found."}), 404

    image_files = glob.glob(os.path.join(training_dir, "*.jpg"))

    # Host the image files and generate URLs
    image_urls = []
    for image_file in image_files:
        filename = os.path.basename(image_file)
        image_url = url_for(
            "get_training_image",
            user_id=user_id,
            model_name=model_name,
            train_dir=TRAIN_DIR,
            filename=filename,
            _external=True,
        )
        image_urls.append(image_url)

    return jsonify({"image_urls": image_urls}), 200


@app.route("/get_training_image/<user_id>/<model_name>/<train_dir>/<filename>")
def get_training_image(user_id, model_name, train_dir, filename):
    training_dir = os.path.join(
        yolov5_dir, "runs", "train", user_id, model_name, train_dir
    )
    return send_from_directory(training_dir, filename)


@app.route("/cancel", methods=["GET"])
def cancel():
    process_id = request.args.get("process_id")
    if not process_id:
        return jsonify({"message": "Process ID is required."}), 400
    try:
        os.kill(process_id, signal.SIGTERM)
        return jsonify({"message": "Process paused successfully."}), 200
    except Exception as e:
        return jsonify({"message": f"Error pausing process: {e}"}), 500


@app.route("/train", methods=["GET"])
@jwt_required()
def start_training():
    user_id = get_jwt_identity()
    model_name = request.args.get("model_name")
    if not model_name:
        return (
            jsonify(
                {"message": "Model Name is actually required please provide it 😊."}
            ),
            400,
        )
    try:
        # Start the training process in a separate thread
        thread = threading.Thread(
            target=train_and_output, args=(user_id, 25, 4, model_name)
        )
        thread.start()
        return jsonify({"message": "Training started successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error starting training: {e}"}), 500


def train_and_output(user_id, EPOCHS, BATCH_SIZE, model_name):
    if EPOCHS is None:
        EPOCHS = 25
    if BATCH_SIZE is None:
        BATCH_SIZE = 4
    print(f"Starting training for user {user_id}")
    # Directory to store results
    res_dir_count = len(glob.glob(yolov5_dir + f"/runs/train/{user_id}/*"))
    print(f"Current number of result directories for user {user_id}: {res_dir_count}")
    RES_DIR = f"results_{res_dir_count+1}"
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
        f"{user_id}/{model_name}/{RES_DIR}",
        "--device",
        "0",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Print the output of the subprocess in the console
    for line in process.stdout:
        print(line.strip())

    # Wait for training to finish
    process.communicate()

    # Return response immediately
    message = f"Training started with {EPOCHS} epochs and {BATCH_SIZE} batch size for user {user_id} in {RES_DIR}."
    return jsonify({"message": message}), 200


# register endpoint
@app.route("/register", methods=["POST"])
def register():
    username = request.json.get("username")
    password = request.json.get("password")
    firstname = request.json.get("firstname")
    lastname = request.json.get("lastname")
    phonenum = request.json.get("phonenum")
    email = request.json.get("email")

    if not username or not password:
        return jsonify({"msg": "Username and password are required."}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists."}), 400

    user = User(username, password, firstname, lastname, phonenum, email)
    db.session.add(user)
    db.session.commit()

    return jsonify({"msg": "Registration successful.", "user_id": user.user_id}), 201


# login endpoint
@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({"msg": "Invalid username or password."}), 401
    access_token = create_access_token(identity=user.user_id, expires_delta=False)
    return jsonify({"access_token": access_token, "user_id": user.user_id}), 200


@app.route("/models", methods=["GET"])
@jwt_required()
def get_models():
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"message": "User ID is required."}), 400

    models = Model.query.filter_by(user_id=user_id).all()
    model_list = []

    for model in models:
        model_name = model.model_name
        model_id = model.model_id
        result = {
            "model_id": model_id,
            "model_name": model_name,
        }
        model_list.append(result)

    return jsonify({"models": model_list}), 200


@app.route("/models/<model_id>", methods=["DELETE"])
@jwt_required()
def delete_model(model_id):
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"message": "User ID is required."}), 400

    # Find the model by model_id and user_id
    model = Model.query.filter_by(user_id=user_id, model_id=model_id).first()
    if not model:
        return jsonify({"message": "Model not found."}), 404

    # Delete the model and related data
    db.session.delete(model)
    db.session.commit()

    # Delete the model's training data folder (assuming it's in the "train" directory)
    train_dir = os.path.join(yolov5_dir, "runs", "train", user_id, model.model_name)
    shutil.rmtree(train_dir, ignore_errors=True)

    return jsonify({"message": "Model deleted successfully."}), 200


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers[
        "Access-Control-Allow-Headers"
    ] = "Content-Type,Authorization,user_id,model_name"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    return response


if __name__ == "__main__":
    app.run(port=5000, debug=True)
