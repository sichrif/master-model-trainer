import json
import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import subprocess
import asyncio
import websockets
import yaml

np.random.seed(42)

TRAIN = True
# Number of epochs to train for.
EPOCHS = 25

with open("data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

class_names = data["names"]
print("Class names: ", class_names)

colors = np.random.uniform(0, 255, size=(len(class_names), 3))


def set_res_dir():
    # Directory to store results
    res_dir_count = len(glob.glob("runs/train/*"))
    print(f"Current number of result directories: {res_dir_count}")
    if TRAIN:
        RES_DIR = f"results_{res_dir_count+1}"
        print(RES_DIR)
    else:
        RES_DIR = f"results_{res_dir_count}"
    return RES_DIR


# """## Clone YOLOV5 Repository"""
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
os.chdir("yolov5")
# subprocess.run(['py', '-m', 'pip', 'install', '-r', 'requirements.txt'])
RES_DIR = set_res_dir()


async def train_and_output(websocket, path):
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

    # Send message to WebSocket when training is done
    message = f"Training is done! {EPOCHS} epochs were completed in {RES_DIR}."
    await websocket.send(message)


async def main():
    async with websockets.serve(train_and_output, "localhost", 8765, ping_timeout=None):
        await asyncio.Future()  # run forever


if TRAIN:
    asyncio.run(main())


def show_valid_results(RES_DIR):
    # subprocess.run(["ls", f"runs/train/{RES_DIR}"])
    EXP_PATH = f"runs/train/{RES_DIR}"
    validation_pred_images = glob.glob(f"{EXP_PATH}/*_pred.jpg")
    print(validation_pred_images)
    for pred_image in validation_pred_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis("off")
        plt.show()


# Helper function for inference on images.
def inference(RES_DIR):
    # Directory to store inference results.
    infer_dir_count = len(glob.glob("runs/detect/*"))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    # Capture video stream from webcam.
    cap = cv2.VideoCapture(0)
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
        ]
    )
    # Release video stream.
    cap.release()
    return INFER_DIR


def visualize(INFER_DIR):
    # Visualize inference images.
    INFER_PATH = f"runs/detect/{INFER_DIR}"
    infer_images = glob.glob(f"{INFER_PATH}/*.jpg")
    for pred_image in infer_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis("off")
        plt.show()


# show_valid_results(RES_DIR)

# Inference on images.
IMAGE_INFER_DIR = inference(RES_DIR)
visualize(IMAGE_INFER_DIR)
inference(RES_DIR)
