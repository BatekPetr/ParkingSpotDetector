#!/usr/bin/env python

'''
Script for YOLO object detection
================
'''

import argparse
import multiprocessing
from dataclasses import dataclass

import cv2
import os

import imutils
import numpy as np
from lxml.etree import SerialisationError

from ultralytics import YOLO

from pythonProject.camera.fps import FPS
from pythonProject.helpers import image_manipulation


@dataclass
class Detection:
    bbox: np.array
    center: np.array
    cls_id: int
    cls_name: str
    confidence: float


class YOLODetector:

    def __init__(self, path_to_yolo_weights: str, use_multiprocessing=False):
        self.model = YOLO(path_to_yolo_weights)

        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            self.img_pipe, self.detections_pipe = multiprocessing.Pipe()


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
                    [x1, y1, x2, y2] = box.xyxy[0].cpu()
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

            self.draw_detections(image, detections)

            # show the image
            cv2.imshow('Yolo detection', image)
            if name:
                cv2.imwrite(name[:-4] + "_det" + name[-4:], image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detection_process(self, pipe: multiprocessing.Pipe):
        while True:
            if pipe.poll():
                # Get the serialized image from the queue
                command = pipe.recv()
                if isinstance(command, bytes):
                    encoded_image = command
                    # Deserialize the image using cv2.imdecode
                    nparr = np.frombuffer(encoded_image, np.uint8)  # Convert bytes back to NumPy array
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image

                    detections = self.detect_in_image(img)
                    img = self.draw_detections(img, detections)

                    # Serialize and send detection results
                    success, encoded_image = cv2.imencode('.jpg', img)
                    # send image for detection into separate process
                    if success:
                        pipe.send(encoded_image.tobytes())  # Put the byte array into the pipe
                    else:
                        raise SerialisationError

                elif isinstance(command, str):
                    if "stop" == command:
                        break

    def draw_detections(self, image, detections):
        for detection in detections:
            # get the respective colour
            colour = self.get_colors(detection.cls_id)

            # draw the rectangle
            cv2.rectangle(image, np.int32(detection.bbox[0]), np.int32(detection.bbox[1]), colour, 2)

            cv2.circle(image, np.int32(detection.center), 5, (0, 0, 255), -1)

            # put the class name and confidence on the image
            cv2.putText(image, f'{detection.cls_name} {detection.confidence:.2f}', np.int32(detection.bbox[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        return image

    def detect_in_video(self, cam, video_name: str = None):

        if self.use_multiprocessing:
            p = multiprocessing.Process(target=self.detection_process, args=(self.detections_pipe,))
            p.start()

        image = cam.take_img()

        if video_name:
            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID', 'MJPG', 'MP4V', etc.
            h, w = imutils.resize(image, 1024).shape[:2]
            out = cv2.VideoWriter(video_name, fourcc, 20, (w, h))  # Output file, codec, fps, resolution

        fps = FPS().start()

        while cam.is_opened():
            if self.use_multiprocessing:
                success, encoded_image = cv2.imencode('.jpg', image)
                # send image for detection into separate process
                if success:
                    self.img_pipe.send(encoded_image.tobytes())  # Put the byte array into the pipe
                else:
                    raise SerialisationError

                # Get the serialized image from the queue
                from_video_process = self.img_pipe.recv()
                if isinstance(from_video_process, bytes):
                    encoded_image = from_video_process

                    # Deserialize the image using cv2.imdecode
                    nparr = np.frombuffer(encoded_image, np.uint8)  # Convert bytes back to NumPy array
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image
            else:
                detections = self.detect_in_image(image)
                self.draw_detections(image, detections)

            fps.update()
            # show the image
            image = imutils.resize(image, 1024)
            fps.add_to_image(image)

            cv2.imshow('frame', image)
            cv2.waitKey(1)

            if isinstance(out, cv2.VideoWriter):
                out.write(image)

            image = cam.take_img()


        cv2.destroyWindow('frame')
        # Stop FPS tracking
        fps.stop()
        out.release()

        if self.use_multiprocessing:
            self.img_pipe.send("stop")
            self.img_pipe.close()
            p.join()

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

    model = YOLODetector(os.path.join("./NN_models", "CarDetector_YOLOv11s_1024.pt"), use_multiprocessing=False)
    if args.img:    # Perform detection on images
        imgs, img_names = image_manipulation.load_images(args.img)
        model.detect_in_images(imgs, img_names)
    else:           # Perform detection in video
        # Load environment variables
        EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
        EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

        RTSP_URL = os.getenv("RTSP_URL")
        from pythonProject.camera.camera_ezviz import CamEzviz
        cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir="../../imgs/parking_dataset",
                       show_video=False)

        import threading
        t = threading.Thread(target=model.detect_in_video, args=(cam, "LiveDetection.avi"))
        t.daemon = True
        t.start()

        cam.control()
        cam.close()

        t.join()