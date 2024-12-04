#!/usr/bin/env python

'''
Script for image undistortion
================

Use this script for lens undistortion.
'''

import argparse
import glob
import sys

import cv2

from pythonProject.camera.camera_calibration import CamIntrinsics


def undistort_images_from_files(imgs_file_names: str, cam_intrinsics: CamIntrinsics):
    in_imgs_names = []
    in_images = []

    if len(imgs_file_names) > 1:  # Work with supplied images
        # read input images
        for img_name in imgs_file_names:
            img = cv2.imread(img_name)
            if img is None:
                print("can't read image " + img_name)
                sys.exit(-1)
            else:
                in_imgs_names.append(img_name)
                in_images.append(img)

    else:
        for img_name in sorted(glob.glob(imgs_file_names[0] + "*[0-9].jpg")):
            img = cv2.imread(img_name)
            if img is None:
                print("can't read image " + img_name)
                sys.exit(-1)
            else:
                in_imgs_names.append(img_name)
                in_images.append(img)

    undistorted_imgs = cam_intrinsics.undistort(in_images)

    return undistorted_imgs, in_imgs_names


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
    undistorted_imgs, in_image_names = undistort_images_from_files(args.img, cam_intrinsics)

    for img, name in zip(undistorted_imgs, in_image_names):
        cv2.imwrite(name[:-4] + "_rect" + name[-4:], img)
