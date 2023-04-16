import os
import glob
import matplotlib.pyplot as plt
import cv2
import requests
import random
import numpy as np
import zipfile
import json
import os
import subprocess
import asyncio
import websockets

np.random.seed(42)
TRAIN = False
# Number of epochs to train for.
EPOCHS = 25


def download_and_unzip(url, save_path):
    # Download file
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)
    # Unzip file
    with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall()


if not os.path.exists("train"):
    # Download and unzip the Roboflow dataset
    # url = 'https://public.roboflow.com/ds/xKLV14HbTF?key=aJzo7msVta'
    # save_path = 'roboflow.zip'
    # download_and_unzip(url, save_path)
    # os.remove('roboflow.zip')

    dirs = ["train", "valid", "test"]

    for i, dir_name in enumerate(dirs):
        all_image_names = sorted(os.listdir(f"{dir_name}/images/"))
        for j, image_name in enumerate(all_image_names):
            if (j % 2) == 0:
                file_name = image_name.split(".jpg")[0]
                os.remove(f"{dir_name}/images/{image_name}")
                os.remove(f"{dir_name}/labels/{file_name}.txt")


def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, "wb").write(file.content)
    else:
        print("File already present, skipping download...")


class_names = ["0", "1", "10", "11", "2", "3", "5", "6", "7", "8", "9"]
colors = np.random.uniform(0, 255, size=(len(class_names), 3))
# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.


def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)
        width = xmax - xmin
        height = ymax - ymin

        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2,
        )
        font_scale = min(1, max(3, int(w / 500)))
        font_thickness = min(2, max(10, int(w / 50)))

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        # Text width and height
        tw, th = cv2.getTextSize(
            class_name, 0, fontScale=font_scale, thickness=font_thickness
        )[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(
            image,
            p1,
            p2,
            color=colors[class_names.index(class_name)],
            thickness=-1,
        )
        cv2.putText(
            image,
            class_name,
            (xmin + 1, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    return image


# Function to plot images with the bounding boxes.


def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0, num_images - 1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], "r") as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(" ")
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i + 1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis("off")
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()


# Visualize a few training images.
plot(
    image_paths="train/images/*",
    label_paths="train/labels/*",
    num_samples=4,
)


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
    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        await websocket.send(output)


async def main():
    async with websockets.serve(train_and_output, "localhost", 8765):
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


def inference(RES_DIR, data_path):
    # Directory to store inference results.
    infer_dir_count = len(glob.glob("runs/detect/*"))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    print(INFER_DIR)

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
    print(infer_images)
    for pred_image in infer_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis("off")
        plt.show()


show_valid_results(RES_DIR)
download_file(
    "https://learnopencv.s3.us-west-2.amazonaws.com/yolov5_inference_data.zip",
    "inference_data.zip",
)
if not os.path.exists("inference_images"):
    subprocess.run(["unzip", "-q", "inference_data.zip"])
else:
    print("Dataset already present")
# Inference on images.
IMAGE_INFER_DIR = inference(RES_DIR, "inference_images")

visualize(IMAGE_INFER_DIR)
inference(RES_DIR, "inference_videos")
# subprocess.Popen('tensorboard --logdir runs/train', shell=True)
