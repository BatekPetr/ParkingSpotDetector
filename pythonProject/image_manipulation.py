import sys
import typing

import cv2
import glob
import matplotlib.pyplot as plt


def display_sample_images(images):

    fig, ax = plt.subplots(nrows=int(len(images)/2) + len(images)%2, ncols=2, figsize=(20, 15))
    axes = ax.flat
    for idx in range(len(images)):
        axes[idx].imshow(images[idx])
        axes[idx].axis('off')

    plt.show()


def load_images(imgs_file_names: list[str]):
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

    return in_images, in_imgs_names
