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


def detect_cars_in_image_process(q_in: queue.Queue, q_out: queue.Queue):
    """Process for NN detection of cars in images.

    :param q_in: Queue for input images
    :param q_out: Queue for output tuples of (image_index, list[detected cars box centers]).
    """
    detector = YOLODetector(os.path.join("./NN_models", "YOLOv11s_MyDataset_imgsz1024.pt"))
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
    """Thread for stitching images into panorama.

    :param q_in: Queue for input images.
    :param q_out: Queue for output of original images and final panorama with transforms
    :param stitcher: Instance of Stitcher object
    """

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

def set_up_threads_and_queues(undistort = True, cam: CamEzviz = None, stitcher = None,
                              detected_queue: queue.Queue = None):
    """Sets up threads, processes and queues.

    :param undistort: Boolean indicating, whether to perform image undistorion
    :param cam: CamEzviz instance implementing undistortion_thread ToDo: move undistortion_thread to CamIntrinsics
    :param stitcher: Instance of Stitcher object
    :param detected_queue: OUTput queue reference for cars detections
    :return: queues: list of created queues
    :return: threads: list of created threads and processes
    :return: out_imgs: list of original images
    :return: pano_queue: reference to OUTput queue of stitching thread
    """

    # Define Queues for process connections
    img_queue = queue.Queue()
    undistorted_queue: queue.Queue = None
    stitching_queue: queue.Queue = None
    pano_queue: queue.Queue = None
    detection_queue = multiprocessing.Queue()
    out_imgs = []

    queues = [img_queue]
    threads = []

    if undistort and cam is not None:
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
        """Manages queues between threads and processes. Selects undistorted images as input for stitching and detection
        if undistorted queue is provided. If not, original images are used.

        Performs image serialization for detection process.

        :param img_queue: queue with original images
        :param undistorted_queue: OPTIONAL queue with undistorted images
        :param stitching_queue: input queue for stitching
        :param detection_queue: input queue for detection
        :param out_imgs: output list of original images
        """
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


def stitch_detect(imgs: list[np.ndarray], stitcher: MyStitcher = None,
                  detected_queue: queue.Queue = None) -> tuple[
        CVImageWithKeypoints, list[np.ndarray]]:
    """
    Stitches images into panorama and detects cars in individual images.

    Multithreaded and multiprocess implementation for performance improvement.

    :param imgs: List of input UNDISTORTED images.
    :param stitcher: Instance of Stitcher class
    :param detected_queue: reference to queue for OUTput of car detections
    :return: CVImageWithKeypoints panorama with keypoints
    :return: arr[np.ndarray] transformations of individual images to the panorama
    """

    undistort = False
    queues, threads, out_imgs, pano_queue = set_up_threads_and_queues(undistort, None, stitcher, detected_queue)
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

    return pano_with_keypoints, transforms


def scan_stitch_detect(cam: CamEzviz, positions: list[(np.float32, np.float32)],
                       name: typing.Union[str, None] = None, path: os.PathLike[any] = None,
                       undistort = True, stitcher: MyStitcher = None, detected_queue: queue.Queue = None) -> tuple[
        CVImageWithKeypoints, list[np.ndarray]]:
    """
    Takes scan of the scene, stitches images into panorama and detects cars in individual images.

    Multithreaded and multiprocess implementation for performance improvement.

    :param cam: Instance of camera with PTZ control
    :param positions: Positions of camera for taking images
    :param name: Name of the images to be taken. If None, images are not saved.
    :param path: Path to save the images
    :param undistort: Whether to undistort images
    :param stitcher: Instance of Stitcher class
    :param detected_queue: reference to queue for OUTput of car detections
    :return: CVImageWithKeypoints panorama with keypoints
    :return: arr[np.ndarray] transformations of individual images to the panorama
    """

    queues, threads, out_imgs, pano_queue = set_up_threads_and_queues(undistort, cam, stitcher, detected_queue)
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

    return pano_with_keypoints, transforms


def get_detected_cars(detected_queue: multiprocessing.Queue,
                      th_pano: np.ndarray, pano_transforms: list[np.ndarray]) -> np.ndarray:
    """Get detected car's positions from detected_queue and transform their coordinates to aligned_pano.

    :param detected_queue: Queue with detected cars
    :param th_pano: Homography from created panorama to the parkslots template coordinates
    :param pano_transforms: list of Homographies from input images to created panorama
    :return: np.ndarray with detected car's center points
    """
    detected_cars = np.empty((1, 2), dtype=np.float32)
    while not detected_queue.empty():
        idx, detections = detected_queue.get()
        th = th_pano.dot(pano_transforms[idx])
        detected_cars = np.vstack([np.squeeze(cv2.perspectiveTransform(np.float32([detections]), th)), detected_cars])

    return detected_cars


def draw_over_pano(pano: np.ndarray, detected_cars: np.ndarray) -> np.ndarray:
    """Draws parking slots detections over panorama.

    :param pano: panorama aligned to parking_slots template coordinates.
    :param detected_cars: array of detected car's center points.
    :return: cv2.Mat image of pano_with_parking_slots
    """
    t_start = time.perf_counter()

    parking_slots = []
    with open("./stitching/parking_slots.json", "r") as f:
        json_data = json.load(f)
    for d in json_data:
        # cv2.polylines(parkslots, [np.int32([d["points"]])], True, (0, 0, 255), 5)
        parking_slots.append(np.int32([d["points"]]))

    parking_slots_pano = pano.copy()
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
    cv2.addWeighted(parking_slots_pano, alpha, pano, 1 - alpha, 0, pano)
    print("Free parking slots detection time: ", time.perf_counter() - t_start)

    # Apply parking slots pano mask to crop resulting image
    with open("./stitching/pano_mask.json", "r") as f:
        json_data = json.load(f)
        mask_points = json_data[0]["points"]

    pano_mask = np.zeros(pano.shape, dtype=np.uint8)
    cv2.fillPoly(pano_mask, np.array([mask_points]), (255, 255, 255))

    pano_with_parking_slots = cv2.bitwise_and(pano, pano_mask)

    return pano_with_parking_slots


def align_pano(pano_with_keypoints: CVImageWithKeypoints, my_stitcher: MyStitcher) -> tuple[np.ndarray, np.ndarray]:
    """Align Panorama with Parkslots template.

    :param pano_with_keypoints: panorama with keypoints
    :param my_stitcher: Stitcher class instance
    :return: cv2.Mat image of aligned panorama
    :return: np.ndarray homography from pano_with_keypoints to aligned_pano
    """
    t_start = time.perf_counter()

    # Load parkslot template with parking slots
    with open("parkslots_pano.pickle", "rb") as handle:
        parkslots, parkslots_transforms, parkslots_keypoints, parkslots_descriptors = pickle.load(handle)
        parkslots_keypoints = list_to_keypoints(parkslots_keypoints)

    matches = my_stitcher.matching(parkslots_descriptors, pano_with_keypoints.descriptors)
    homography, mask = find_homography(parkslots_keypoints, pano_with_keypoints.keypoints, matches)

    th_pano = homography
    h, w = parkslots.shape[:2]
    aligned_pano = cv2.warpPerspective(pano_with_keypoints.img, np.float32(th_pano), (w, h))
    print(f"Alignement Time: {time.perf_counter() - t_start}")

    return aligned_pano, th_pano


def crop_image_borders(img: np.ndarray):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask (invert to make the border black)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for the largest contour (non-black area)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image
    cropped_image = img[y:y + h, x:x + w]

    return cropped_image


def main(args):
    scan_start = time.perf_counter()

    keypoint_features = KeypointFeatures.SIFT
    my_stitcher = MyStitcher(keypoint_features)

    # Create a queue for detections
    detected_queue = multiprocessing.Queue()

    if args.img:  # Perform detection on images
        print("Loading images from disk ...")
        imgs, img_names = image_manipulation.load_images(args.img)

        print("Preprocessing images ...")
        imgs = preprocess_images(imgs)

        print("Stitching and detection ...")
        pano_with_keypoints, pano_transforms = stitch_detect(imgs, stitcher=my_stitcher, detected_queue=detected_queue)
    else:  # Perform detection on a new camera scan
        # Load environment variables
        EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
        EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

        RTSP_URL = os.getenv("RTSP_URL")

        cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir="../imgs/testing",
                       show_video=True, use_multiprocessing=False)
        print("Camera scan, stitching and detection ...")
        pano_with_keypoints, pano_transforms = scan_stitch_detect(cam, positions=[(0.5, 0), (0.4, 0), (0.62, 0)],
                                                                  # (0.5, 6)],
                                                                  name=args.save_name, undistort=True,
                                                                  stitcher=my_stitcher,
                                                                  detected_queue=detected_queue)
        cam.close()

    aligned_pano, th_pano = align_pano(pano_with_keypoints, my_stitcher)

    detected_cars = get_detected_cars(detected_queue, th_pano, pano_transforms)

    aligned_pano_with_parking_slots = draw_over_pano(aligned_pano, detected_cars)

    # Crop resulting pano
    cropped_pano_with_parking_slots = crop_image_borders(aligned_pano_with_parking_slots)

    print(f"Total time: {time.perf_counter() - scan_start}")
    cv2.imshow("Detections in ALIGNED", imutils.resize(cropped_pano_with_parking_slots, width=1924))

    if args.img is not None:
        cv2.imwrite(args.img[0] + "_pano.jpg", cropped_pano_with_parking_slots)

    cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='stitch_and_detect.py',
                                     description='Stitch images (either loaded from files or after taking by camera) '
                                                 'and perform available parking slot detection.')
    parser.add_argument('--save_name', nargs='?', type=str, default=None)
    parser.add_argument('--img', nargs='+', help='input images')

    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    args = parser.parse_args()

    main(args)





