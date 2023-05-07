import subprocess

from flask import jsonify


def train_and_output(EPOCHS, BATCH_SIZE, yolov5_dir, RES_DIR):
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
