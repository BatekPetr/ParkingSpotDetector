#!/usr/bin/env python

'''
Script for image undistortion
================

Use this script for lens undistortion.
'''

import argparse
import glob
import sys
import typing

import cv2

from pythonProject import image_manipulation
from pythonProject.camera.camera_calibration import CamIntrinsics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='undistort_imgs.py',
                                     description='Undistort images from disk using camera parameters.')
    parser.add_argument('--img', nargs='+', required=True, help='input images')
    parser.add_argument('--output',
                        help='Save resulting image as.')


    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    args = parser.parse_args()

    # Load camera parameters for image rectification
    cam_intrinsics = CamIntrinsics("./Ezviz_C6N")
    in_images, in_image_names = image_manipulation.load_images(args.img)
    undistorted_imgs = cam_intrinsics.undistort(in_images)

    for img, name in zip(undistorted_imgs, in_image_names):
        cv2.imwrite(name[:-4] + "_rect" + name[-4:], img)
