import ultralytics
import os
import json
import numpy as np
import cv2
import random
import shutil

if __name__ == "__main__":
    with open('New_Code/configs/preprocessing_config.json') as f:
        preprocessing_config = json.load(f)

    yolo_path = preprocessing_config["yolo_path"]
    model = ultralytics.YOLO('yolo11m.pt')
    model.train(data = os.path.join(yolo_path, "data.yaml"), epochs = 50, imgsz = 640, project = yolo_path, name = "attempt_1")


