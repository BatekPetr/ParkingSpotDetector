#!/usr/bin/env python

'''
Stitching sample
================

Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

# Python 2/3 compatibility
from __future__ import print_function

import os
import time

import cv2 as cv

import argparse
import sys

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.camera.camera_ezviz import CamEzviz


def undistort(in_images):

    # Load camera parameters for image rectification
    cam_instrinsics = CamIntrinsics("../Ezviz_C6N")

    # Undistort images
    out_images = []
    for img in in_images:
        # Image undistortion
        dst = cam_instrinsics.undistort(img)

        out_images.append(dst)

    return out_images


def stitch(in_images, mode):
    # ![stitching]
    stitcher = cv.Stitcher.create(mode)
    status, pano = stitcher.stitch(in_images)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)

    print("Stitching completed successfully.")
    cv.imshow("Stitched image", pano)

    return pano


if __name__ == '__main__':
    modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

    parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
    parser.add_argument('--mode',
                        type=int, choices=modes, default=cv.Stitcher_PANORAMA,
                        help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                             'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                             'for stitching materials under affine transformation, such as scans.' % modes)
    parser.add_argument('--output',
                        help='Save resulting image as.')
    parser.add_argument('--img', nargs='+', help='input images')

    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    t1 = time.time_ns()
    args = parser.parse_args()

    if args.img:    # Work with supplied images
        # read input images
        in_images = []
        for img_name in args.img:
            img = cv.imread(img_name)
            if img is None:
                print("can't read image " + img_name)
                sys.exit(-1)
            in_images.append(img)
        preprocessed_imgs = undistort(in_images)
    else:   # No images supplied. Take scan with camera.
        # Load environment variables
        EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
        EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

        RTSP_URL = os.getenv("RTSP_URL")

        cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD)
        t1 = time.time_ns()
        images = cam.scan(undistort=False)
        preprocessed_imgs = cam.undistort(images)
        cam.close()

    stitched = stitch(in_images=preprocessed_imgs, mode=args.mode)

    if args.output:
        cv.imwrite(args.output, stitched)

    print("time: " + str(1e-9 * (time.time_ns() - t1)))

    cv.waitKey(0)
    cv.destroyAllWindows()