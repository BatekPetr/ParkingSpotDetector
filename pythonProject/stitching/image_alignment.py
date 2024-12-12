import json
import os
import pickle
import queue
import threading
import time

import cv2
import imutils
import numpy as np
import multiprocessing

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.detection.YOLO_detection import YOLODetector
from pythonProject.image_manipulation import load_images
from pythonProject.stitching.my_stitching import (preprocess_images, find_homography, concatenate_images,
                                                  Matching, MyStitcher, keypoints_to_list, list_to_keypoints)


def detect_cars_in_image(img, idx = None, queue = None):
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


if __name__ == "__main__":
    matching = Matching.SIFT
    my_stitcher = MyStitcher(matching)
    # # Create panorama from images
    # ref_imgs, _ = load_images(["../imgs/pano/template"])
    # ref_imgs = preprocess_images(ref_imgs)
    # parkslots, parkslots_transforms, parkslots_keypoints, parkslots_descriptors = my_stitcher.stitch([img.copy() for img in ref_imgs])
    # # cv2.imwrite("parkslots_pano.jpg", parkslots)
    # with open("parkslots_pano.pickle", "wb") as handle:
    #      pickle.dump((parkslots, parkslots_transforms, keypoints_to_list(parkslots_keypoints), parkslots_descriptors), handle)
    with open("parkslots_pano.pickle", "rb") as handle:
         parkslots, parkslots_transforms, parkslots_keypoints, parkslots_descriptors = pickle.load(handle)
         parkslots_keypoints = list_to_keypoints(parkslots_keypoints)


    # cv2.imwrite("parkslots_pano_det.jpg", parkslots)

    imgs, _ = load_images(["../imgs/pano/test_afternoon"])
    imgs = preprocess_images(imgs )
    imgs[0], imgs[1] = imgs[1], imgs[0]

    show_start = time.perf_counter()

    # Detect cars in images
    print("Starting car detection.")
    t_start = time.perf_counter()

    detected_cars = {}
    threads = []
    # Create a queue
    queue = queue.Queue()
    for idx, img in enumerate(imgs):
        detected_cars[idx] = []
        t = threading.Thread(target=detect_cars_in_image, args=(img.copy(), idx, queue))
        threads.append(t)
        t.start()

    # print(f"Detection Time: {time.perf_counter() - t_start}")
    print("Starting stitching.")
    t_start = time.perf_counter()
    pano, pano_transforms, pano_keypoints, pano_descriptors = my_stitcher.stitch(imgs)
    print(f"Stitching Time: {time.perf_counter() - t_start}")

    t_start = time.perf_counter()
    matches = my_stitcher.matching(parkslots_descriptors, pano_descriptors)
    homography, mask = find_homography(parkslots_keypoints, pano_keypoints, matches)
    # _, th_parkslots, th_pano = concatenate_images(parkslots, pano, homography)
    th_parkslots = np.eye(3, 3, dtype=np.float32)
    th_pano = homography
    h, w = parkslots.shape[:2]
    aligned_pano = cv2.warpPerspective(pano, np.float32(th_pano), (w, h))
    print(f"Alignement Time: {time.perf_counter() - t_start}")

    parking_slots = []
    with open("./parking_slots.json", "r") as f:
        json_data = json.load(f)
    for d in json_data:
        #cv2.polylines(parkslots, [np.int32([d["points"]])], True, (0, 0, 255), 5)

        parking_slots.append(np.int32(cv2.perspectiveTransform(np.float32([d["points"]]), th_parkslots)))
        # cv2.polylines(aligned_pano, [parking_slots[-1]],
        #               True, (0, 0, 255), 5)

    # Wait for all processes to finish
    for t in threads:
        t.join()

    detected_cars = np.empty((1, 2), dtype=np.float32)
    while not queue.empty():
        idx, detections = queue.get()
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

    with open("./pano_mask.json", "r") as f:
        json_data = json.load(f)
        mask_points = json_data[0]["points"]

    pano_mask = np.zeros(aligned_pano.shape, dtype=np.uint8)
    cv2.fillPoly(pano_mask, np.array([mask_points]), (255, 255, 255))
    aligned_pano = cv2.bitwise_and(aligned_pano, pano_mask)

    # intrinsics = CamIntrinsics(os.path.join("./camera", "Ezviz_C6N"))
    # aligned_pano = intrinsics.cylindrical_warp(aligned_pano)
    print(f"Total time: {time.perf_counter() - show_start}")
    cv2.imshow("Detections in ALIGNED", imutils.resize(aligned_pano, width=1924))
    # cv2.imwrite("results.jpg", aligned_pano)
    cv2.waitKey()



