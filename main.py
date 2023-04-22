import os
from flask import Flask, request, jsonify
import zipfile
import json
import glob
import subprocess
import websockets
import asyncio

app = Flask(__name__)


def set_res_dir():
    subprocess.run(["cd", "yolov5"], shell=True)
    # Directory to store results
    res_dir_count = len(glob.glob("runs/train/*"))
    print(f"Current number of result directories: {res_dir_count}")
    RES_DIR = f"results_{res_dir_count}"
    return RES_DIR


async def train_and_output(websocket, EPOCHS, BATCH_SIZE):
    os.chdir("yolov5")
    if EPOCHS is None:
        EPOCHS = 25
    if BATCH_SIZE is None:
        BATCH_SIZE = 4
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
        BATCH_SIZE,
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

    # Send message to WebSocket when training is done
    message = f"Training is done! {EPOCHS} epochs were completed in {RES_DIR}."
    await websocket.send(message)


async def main():
    async with websockets.serve(
        train_and_output(25, 4), "localhost", 8765, ping_timeout=None
    ):
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
def start_detection():
    # Directory to store inference results.
    infer_dir_count = len(glob.glob("runs/detect/*"))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    # Inference on video stream.
    subprocess.run(
        [
            "python",
            "detect.py",
            "--weights",
            f"runs/train/{RES_DIR}/weights/best.pt",
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
    return f"Detection job started in directory: {INFER_DIR}"


@app.route("/train", methods=["GET"])
def start_websocket():
    try:
        asyncio.run(main())
        return jsonify({"message": "WebSocket started successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error starting WebSocket: {e}"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
