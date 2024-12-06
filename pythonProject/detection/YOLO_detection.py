#!/usr/bin/env python

'''
Script for YOLO object detection
================
'''

import argparse
from dataclasses import dataclass

import cv2
import os

import numpy as np
from ultralytics import YOLO

from pythonProject import image_manipulation


@dataclass
class Detection:
    bbox: np.array
    center: np.array
    cls_id: int
    cls_name: str
    confidence: float


class YOLODetector:

    def __init__(self, path_to_yolo_weights: str):
        self.model = YOLO(path_to_yolo_weights)


    def detect_in_image(self, image: np.ndarray, confidence=0.1):
        results = self.model(image)
        detections = []

        for result in results:
            # iterate over each box
            for box in result.boxes:
                box_conf = box.conf[0]
                # check if confidence is greater than 40 percent
                if box_conf > confidence:
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    bbox = np.array([(x1, y1), (x2, y2)])
                    # Find center of box detection
                    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                    # get the class
                    cls_id = int(box.cls[0])

                    # get the class name
                    cls_name = result.names[cls_id]

                    detections.append(Detection(bbox, center, cls_id, cls_name, box_conf))

        return detections

    def detect_in_images(self, images: list[np.ndarray], names: list[str] = None):
        if names is None:
            names = [None] * len(images)

        for image, name in zip(images, names):
            detections = self.detect_in_image(image, confidence=0.1)

            for detection in detections:
                # get the respective colour
                colour = self.get_colors(detection.cls_id)

                # draw the rectangle
                cv2.rectangle(image, detection.bbox[0], detection.bbox[1], colour, 2)
                cv2.circle(image, detection.center, 5, (0, 0, 255), -1)

                # put the class name and confidence on the image
                cv2.putText(image, f'{detection.cls_name} {detection.confidence:.2f}', detection.bbox[0],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                # show the image
            cv2.imshow('Yolo detection', image)
            if name:
                cv2.imwrite(name[:-4] + "_det" + name[-4:], image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_in_video(self, cam):

        image = cam.take_img()

        while image is not None:
            results = self.model(image)

            for result in results:
                # get the classes names
                classes_names = result.names

                # iterate over each box
                for box in result.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.1:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = classes_names[cls]

                        # get the respective colour
                        colour = self.get_colors(cls)

                        # draw the rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

                        # put the class name and confidence on the image
                        cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                # show the image
            cv2.imshow('frame', cv2.resize(image, (960, 540)))
            image = cam.take_img()

        cv2.destroyWindow('frame')


    # Function to get class colors
    def get_colors(self, cls_num):
        # cls_num = len(self.model.names)
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] *
        (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)



if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='YOLO_detection.py',
                                     description='Use NN model to detect objects in images or video. Supply image names'
                                                 'or Unix style pathname pattern/')
    parser.add_argument('--img', nargs='+', help='input images')


    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    args = parser.parse_args()

    model = YOLODetector(os.path.join("./NN_models", "YOLOv11x_MyDataset_imgsz1024.pt"))
    if args.img:    # Perform detection on images
        imgs, img_names = image_manipulation.load_images(args.img)
        model.detect_in_images(imgs, img_names)
    else:           # Perform detection in video
        # Load environment variables
        EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
        EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

        RTSP_URL = os.getenv("RTSP_URL")
        from pythonProject.camera.camera_ezviz import CamEzviz
        cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir="../../imgs/parking_dataset", show_video=False)

        import threading
        t = threading.Thread(target=model.detect_in_video, args=(cam,))
        t.daemon = True
        t.start()

        cam.control()
        cam.close()