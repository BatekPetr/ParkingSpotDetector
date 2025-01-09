#!/usr/bin/env python

'''
Script for panorama creation and parking slot detection.
================
'''

import argparse
import json
import os
import pickle
import queue
import threading
import time
import typing

import cv2
import imutils
import numpy as np
import multiprocessing

from lxml.etree import SerialisationError

from pythonProject.helpers import image_manipulation
from pythonProject.camera.camera_ezviz import CamEzviz
from pythonProject.detection.YOLO_detection import YOLODetector
from pythonProject.stitching.my_stitching import (preprocess_images, find_homography, KeypointFeatures, MyStitcher,
                                                  list_to_keypoints, CVImageWithKeypoints)


def detect_cars_in_image_thread(img, idx = None, queue = None):
    detector = YOLODetector(os.path.join("./NN_models", "YOLOv11x_MyDataset_imgsz1024.pt"))
    detections = detector.detect_in_image(img, 0.1)
    detected_cars = []
    for detection in detections:
        if detection.cls_name == "car" or detection.cls_name == "truck":
            # detection.bbox = np.squeeze(cv2.perspectiveTransform(np.array([detection.bbox]), th))
            # detection.center = np.squeeze(cv2.perspectiveTransform(np.array([[detection.center]]), th))
            detected_cars.append(detection.center)

    if queue:
        queue.put( (idx, detected_cars) )

    return detected_cars

def detect_cars_in_image_process(q_in: queue.Queue, q_out: queue.Queue):
    detector = YOLODetector(os.path.join("./NN_models", "YOLOv11x_MyDataset_imgsz1024.pt"))
    idx = 0
    while True:
        encoded_image = q_in.get()
        if encoded_image is None:
            break
        else:
            # Deserialize the image using cv2.imdecode
            nparr = np.frombuffer(encoded_image, np.uint8)  # Convert bytes back to NumPy array
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image

        detections = detector.detect_in_image(img, 0.1)
        detected_cars = []
        for detection in detections:
            if detection.cls_name == "car" or detection.cls_name == "truck":
                detected_cars.append(detection.center)

        q_out.put((idx, detected_cars))
        idx += 1

    print("Quitting detect_cars_in_image_process.")


def stitching_thread(q_in: queue.Queue, q_out: queue.Queue, stitcher: MyStitcher):
    ref_img = q_in.get()
    q_out.put(ref_img)
    transforms = None

    while True:
        img = q_in.get()
        if img is None:
            q_out.put((ref_img, transforms))
            break
        q_out.put(img)
        ref_img, transforms = stitcher.stitch([ref_img, img]) # ref_img becomes CVImageWithKeypoints

    print("Quitting stitching_thread.")

def set_up_threads_and_queues(undistort = None, stitcher = None, detected_queue: queue.Queue = None):
    # Define Queues for process connections
    img_queue = queue.Queue()
    undistorted_queue: queue.Queue = None
    stitching_queue: queue.Queue = None
    pano_queue: queue.Queue = None
    detection_queue = multiprocessing.Queue()
    out_imgs = []

    queues = [img_queue]
    threads = []

    if undistort:
        undistortion_queue = img_queue
        undistorted_queue = queue.Queue()
        undistort_thread = threading.Thread(target=cam.undistortion_thread, args=(undistortion_queue,
                                                                                  undistorted_queue))
        undistort_thread.daemon = True
        undistort_thread.start()

        queues += [undistorted_queue]
        threads.append(undistort_thread)

    if stitcher is not None:
        stitching_queue = queue.Queue()
        pano_queue = queue.Queue()
        stitch_thread = threading.Thread(target=stitching_thread,
                                         args=(stitching_queue, pano_queue, stitcher))
        stitch_thread.daemon = True
        stitch_thread.start()

        queues += [stitching_queue, pano_queue]
        threads.append(stitch_thread)

    if detected_queue is not None:
        detection_process = multiprocessing.Process(target=detect_cars_in_image_process, args=(detection_queue,
                                                                                               detected_queue))
        detection_process.daemon = True
        detection_process.start()

        queues += [detection_queue]

    def manage_queues(img_queue, undistorted_queue, stitching_queue, detection_queue, out_imgs):
        if undistorted_queue is not None:
            in_queue = undistorted_queue
        else:
            in_queue = img_queue

        while True:
            img = in_queue.get()

            if img is None:
                break
            else:
                out_imgs.append(img)
                stitching_queue.put(img.copy())

                # Serialize and send detection results
                success, encoded_image = cv2.imencode('.jpg', img)
                # send image for detection into separate process
                if success:
                    detection_queue.put(encoded_image.tobytes())  # Put the byte array into the pipe
                else:
                    raise SerialisationError

        print("Quitting manage_queues.")

    queue_management_thread = threading.Thread(target=manage_queues, args=(img_queue, undistorted_queue,
                                                                           stitching_queue, detection_queue,
                                                                           out_imgs))
    queue_management_thread.start()

    threads.append(queue_management_thread)

    return queues, threads, out_imgs, pano_queue


def stitch_detect(imgs: list[np.ndarray], undistort = False, stitcher: MyStitcher = None,
                  detected_queue: queue.Queue = None):

    queues, threads, out_imgs, pano_queue = set_up_threads_and_queues(undistort, stitcher, detected_queue)
    img_queue = queues[0]

    try:
        for img in imgs:
            img_queue.put(img)

        # Let threads and process start spinning
        # ToDo: handle in better way
        time.sleep(10)

        # Finish threads and process by sending None to Queues
        for q in queues:
            q.put(None)

        for t in threads:
            t.join()

        # Wait max 20s to get panorama with keypoints
        pano_with_keypoints, transforms = None, None
        secs = 0
        while pano_with_keypoints is None and secs < 20:
            while not pano_queue.empty():
                item = pano_queue.get()
                if item is not None and isinstance(item[0], CVImageWithKeypoints):
                    pano_with_keypoints, transforms = item
            time.sleep(2)
            secs += 2

    except BaseException as exp:
        print(exp)
        return 1

    return out_imgs, pano_with_keypoints, transforms


def scan_stitch_detect(cam: CamEzviz, positions: list[(np.float32, np.float32)],
                       name: typing.Union[str, None] = None, path: os.PathLike[any] = None,
                       undistort = True, stitcher: MyStitcher = None, detected_queue: queue.Queue = None):
    """
    Take scan of the scene. Camera rotates and takes multiple pictures.

    :param name: Name of the images to be taken. If None, images are not saved.
    :param path: Path to save the images
    :param undistort: Whether to undistort images
    :return: arr[cv.Mat] images
    """

    queues, threads, out_imgs, pano_queue = set_up_threads_and_queues(undistort, stitcher, detected_queue)
    img_queue = queues[0]

    if path is None:
        path = cam.img_save_dir

    if name:
        img_path_name = os.path.join(path, name)
    else:
        img_path_name = None

    try:
        for i, pos in enumerate(positions):
            # Vertical movement to given coordinate is not supported.
            # Must improvize for vertical movements.
            for _ in range(pos[1]):
                cam.camera.move("up")
                time.sleep(2)

            # Movement in X direction
            cam.camera.move_coordinates(*pos)
            time.sleep(10)

            img = cam.take_img(img_path_name, f"_{i}.jpg")
            img_queue.put(img)

            # return back to previous Y position
            for _ in range(pos[1]):
                cam.camera.move("down")
                time.sleep(2)

        # Return camera to initial position
        cam.camera.move_coordinates(0.5, y_axis=0.0)

        # Finish threads and process by sending None to Queues
        for q in queues:
            q.put(None)

        for t in threads:
            t.join()

        # Wait max 20s to get panorama with keypoints
        pano_with_keypoints, transforms = None, None
        secs = 0
        while pano_with_keypoints is None and secs < 20:
            while not pano_queue.empty():
                item = pano_queue.get()
                if item is not None and isinstance(item[0], CVImageWithKeypoints):
                    pano_with_keypoints, transforms = item
            time.sleep(2)
            secs += 2

    except BaseException as exp:
        print(exp)
        return 1

    return out_imgs, pano_with_keypoints, transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='stitch_and_detect.py',
                                     description='Stitch images (either loaded from files or after taking by camera) '
                                                 'and perform available parking slot detection.')
    parser.add_argument('--img', nargs='+', help='input images')

    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    args = parser.parse_args()

    scan_start = time.perf_counter()

    keypoint_features = KeypointFeatures.SIFT
    my_stitcher = MyStitcher(keypoint_features)

    # Load parkslot template with parking slots
    with open("parkslots_pano.pickle", "rb") as handle:
         parkslots, parkslots_transforms, parkslots_keypoints, parkslots_descriptors = pickle.load(handle)
         parkslots_keypoints = list_to_keypoints(parkslots_keypoints)

    detected_cars = {}
    threads = []
    # Create a queue for detections
    detected_queue = multiprocessing.Queue()

    if args.img:    # Perform detection on images
        print("Loading images from disk.")
        imgs, img_names = image_manipulation.load_images(args.img)
        imgs = preprocess_images(imgs)

        imgs, pano_with_keypoints, pano_transforms = stitch_detect(imgs, undistort=False,
                                                                        stitcher=my_stitcher,
                                                                        detected_queue=detected_queue)
    else:       # Perform detection on a new camera scan
        # Load environment variables
        EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
        EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

        RTSP_URL = os.getenv("RTSP_URL")

        cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir="../imgs/testing",
                       show_video=True, use_multiprocessing=False)
        print("Starting camera scan.")
        imgs, pano_with_keypoints, pano_transforms = scan_stitch_detect(cam, positions=[(0.5, 0), (0.4, 0), (0.62, 0)], # (0.5, 6)],
                                                                        name="night_demo_1", undistort=True,
                                                                        stitcher=my_stitcher,
                                                                        detected_queue=detected_queue)
        cam.close()

    t_start = time.perf_counter()
    matches = my_stitcher.matching(parkslots_descriptors, pano_with_keypoints.descriptors)
    homography, mask = find_homography(parkslots_keypoints, pano_with_keypoints.keypoints, matches)

    th_parkslots = np.eye(3, 3, dtype=np.float32)
    th_pano = homography
    h, w = parkslots.shape[:2]
    aligned_pano = cv2.warpPerspective(pano_with_keypoints.img, np.float32(th_pano), (w, h))
    print(f"Alignement Time: {time.perf_counter() - t_start}")

    parking_slots = []
    with open("./stitching/parking_slots.json", "r") as f:
        json_data = json.load(f)
    for d in json_data:
        #cv2.polylines(parkslots, [np.int32([d["points"]])], True, (0, 0, 255), 5)

        parking_slots.append(np.int32(cv2.perspectiveTransform(np.float32([d["points"]]), th_parkslots)))

    # Wait for all processes to finish
    for t in threads:
        t.join()

    detected_cars = np.empty((1, 2), dtype=np.float32)
    while not detected_queue.empty():
        idx, detections = detected_queue.get()
        th = th_pano.dot(pano_transforms[idx])
        detected_cars = np.vstack([np.squeeze(cv2.perspectiveTransform(np.float32([detections]), th)), detected_cars])

    t_start = time.perf_counter()
    parking_slots_pano = aligned_pano.copy()
    for parking_slot in parking_slots:
        free = True
        for car in detected_cars:
            if cv2.pointPolygonTest(parking_slot, car, measureDist=False) == 1:
                cv2.fillPoly(parking_slots_pano, parking_slot, color=(0, 0, 255))
                free = False
                break

        # If no car detected in parking_slot
        if free:
            cv2.fillPoly(parking_slots_pano, parking_slot, color=(0, 255, 0))

    alpha = 0.3
    cv2.addWeighted(parking_slots_pano, alpha, aligned_pano, 1 - alpha, 0, aligned_pano)
    print("Free parking slots detection time: ", time.perf_counter() - t_start)

    with open("./stitching/pano_mask.json", "r") as f:
        json_data = json.load(f)
        mask_points = json_data[0]["points"]

    pano_mask = np.zeros(aligned_pano.shape, dtype=np.uint8)
    cv2.fillPoly(pano_mask, np.array([mask_points]), (255, 255, 255))
    aligned_pano = cv2.bitwise_and(aligned_pano, pano_mask)

    # intrinsics = CamIntrinsics(os.path.join("./camera", "Ezviz_C6N"))
    # aligned_pano = intrinsics.cylindrical_warp(aligned_pano)
    print(f"Total time: {time.perf_counter() - scan_start}")
    cv2.imshow("Detections in ALIGNED", imutils.resize(aligned_pano, width=1924))
    cv2.imwrite("results.jpg", aligned_pano)
    cv2.waitKey()



