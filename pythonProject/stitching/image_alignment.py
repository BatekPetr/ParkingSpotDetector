import time

import cv2
import imutils

from pythonProject.camera.undistort_imgs import load_images
from pythonProject.stitching.my_stitching import (preprocess_images, find_homography, concatenate_images,
                                                  Matching, MyStitcher)

if __name__ == "__main__":
    matching = Matching.ORB
    my_stitcher = MyStitcher(matching)
    # Create panorama from images
    ref_imgs, _ = load_images(["../imgs/pano/template"])
    ref_imgs = preprocess_images(ref_imgs)
    parkslots, parkslots_transforms, parkslots_keypoints, parkslots_descriptors = my_stitcher.stitch(ref_imgs)

    imgs, _ = load_images(["../imgs/distorted"])
    imgs = preprocess_images(imgs )
    imgs[0], imgs[1] = imgs[1], imgs[0]

    t_start = time.perf_counter()
    pano, pano_transforms, pano_keypoints, pano_descriptors = my_stitcher.stitch(imgs)
    print(f"Stitching Time: {time.perf_counter() - t_start}")

    t_start = time.perf_counter()
    matches = my_stitcher.matching(pano_descriptors, parkslots_descriptors)
    homography, mask = find_homography(pano_keypoints, parkslots_keypoints, matches)
    aligned_pano, *_ = concatenate_images(pano, parkslots, homography)
    print(f"Alignement Time: {time.perf_counter() - t_start}")

    cv2.imshow("Aligned", imutils.resize(aligned_pano, 1924))
    cv2.imwrite("aligned_" + str(matching) + ".jpg", aligned_pano)
    cv2.waitKey()



