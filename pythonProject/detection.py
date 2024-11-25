import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os



def display_sample_images(images):

    fig, ax = plt.subplots(nrows=int(len(images)/2) + len(images)%2, ncols=2, figsize=(20, 15))
    axes = ax.flat
    for idx in range(len(images)):
        axes[idx].imshow(images[idx])
        axes[idx].axis('off')

    plt.show()


def load_image(path):
    image = cv2.imread(path)

    # Convert image in BGR format to RGB.
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_images(folder = 'object_detection_images', file_pattern='*.png'):
    image_paths = sorted(glob.glob(os.path.join(folder, file_pattern)))

    for idx in range(len(image_paths)):
        print(image_paths[idx])

    images = []
    for image_path in image_paths:
        image = load_image(image_path)
        images.append(image)

    return images
