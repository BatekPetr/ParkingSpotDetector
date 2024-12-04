#!/usr/bin/env python

'''
Stitching sample
================

Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

# Python 2/3 compatibility
from __future__ import print_function

import glob
import os
import time

import cv2

import argparse
import sys

import imutils

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.camera.camera_ezviz import CamEzviz
from pythonProject.camera.undistort_imgs import undistort_images_from_files
from pythonProject.my_stitching import find_homography_to_base


def stitch(in_images, mode):
    # ![stitching]
    stitcher = cv2.Stitcher.create(mode)
    status, pano = stitcher.stitch(in_images)

    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)

    print("Stitching completed successfully.")
    cv2.imshow("Stitched image", imutils.resize(pano, width=1928))

    return pano


if __name__ == '__main__':
    modes = (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS)

    parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
    parser.add_argument('--mode',
                        type=int, choices=modes, default=cv2.Stitcher_PANORAMA,
                        help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                             'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                             'for stitching materials under affine transformation, such as scans.' % modes)
    parser.add_argument('--output',
                        help='Save resulting image as.')
    parser.add_argument('--img', nargs='+', help='input images')

    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    in_images = []

    t1 = time.time_ns()
    args = parser.parse_args()

    if args.img:
        # Load camera parameters for image rectification
        cam_intrinsics = CamIntrinsics("./camera//Ezviz_C6N")
        preprocessed_imgs, _ = undistort_images_from_files(args.img, cam_intrinsics)
    else:   # No images supplied. Take scan with camera.
        # Load environment variables
        EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
        EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

        RTSP_URL = os.getenv("RTSP_URL")

        cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD)
        t1 = time.time_ns()
        in_images = cam.scan(name=args.output, undistort=False)
        preprocessed_imgs = cam.undistort(in_images)
        cam.close()

    stitched = stitch(in_images=preprocessed_imgs, mode=args.mode)
    if args.output:
        cv2.imwrite(args.output + "_stitched.jpg", stitched)
    ref_img = cv2.imread("base_rect_old.jpg", cv2.IMREAD_GRAYSCALE)  # Reference image.
    ref_mask = cv2.imread("base_rect_mask.jpg", cv2.IMREAD_GRAYSCALE)  # Reference image.
    homography, mask = find_homography_to_base(ref_img, stitched, ref_mask)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    height, width = ref_img.shape
    transformed_img = cv2.warpPerspective(stitched, homography, (width, height))

    cv2.imshow("Transformed pano", transformed_img)

    print("time: " + str(1e-9 * (time.time_ns() - t1)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()