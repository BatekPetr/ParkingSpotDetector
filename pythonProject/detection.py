import os
import time
from os.path import exists

import numpy as np
import cv2
import zipfile
import requests
import glob as glob

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
import matplotlib.pyplot as plt
import warnings
import logging
import absl

# Filter absl warnings
warnings.filterwarnings("ignore", module="absl")

# Capture all warnings in the logging system
logging.captureWarnings(True)

# Set the absl logger level to 'error' to suppress warnings
absl_logger = logging.getLogger("absl")
absl_logger.setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Download Sample Images
def download_and_unzip_images(url='https://www.dropbox.com/s/h7l1lmhvga6miyo/object_detection_images.zip?dl=1',
                              save_name='object_detection_images.zip'):

    download_file(url, save_name)
    unzip(zip_file='object_detection_images.zip')


def download_file(url, save_name):
    url = url
    file = requests.get(url)

    open(save_name, 'wb').write(file.content)


def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all")
    except:
        print("Invalid file")


## Display Sample Images
def display_sample_images(images):

    fig, ax = plt.subplots(nrows=int(len(images)/2) + len(images)%2, ncols=2, figsize=(20, 15))
    axes = ax.flat
    idx = 0
    for idx in range(len(images)):
        axes[idx].imshow(images[idx][0])
        axes[idx].axis('off')

    plt.show()


def load_image(path):
    image = cv2.imread(path)

    # Convert image in BGR format to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add a batch dimension which is required by the model.
    image = np.expand_dims(image, axis=0)

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


## Define a Dictionary that Maps Class IDs to Class Names

class_index =  \
{
         1: 'person',
         2: 'bicycle',
         3: 'car',
         4: 'motorcycle',
         5: 'airplane',
         6: 'bus',
         7: 'train',
         8: 'truck',
         9: 'boat',
         10: 'traffic light',
         11: 'fire hydrant',
         13: 'stop sign',
         14: 'parking meter',
         15: 'bench',
         16: 'bird',
         17: 'cat',
         18: 'dog',
         19: 'horse',
         20: 'sheep',
         21: 'cow',
         22: 'elephant',
         23: 'bear',
         24: 'zebra',
         25: 'giraffe',
         27: 'backpack',
         28: 'umbrella',
         31: 'handbag',
         32: 'tie',
         33: 'suitcase',
         34: 'frisbee',
         35: 'skis',
         36: 'snowboard',
         37: 'sports ball',
         38: 'kite',
         39: 'baseball bat',
         40: 'baseball glove',
         41: 'skateboard',
         42: 'surfboard',
         43: 'tennis racket',
         44: 'bottle',
         46: 'wine glass',
         47: 'cup',
         48: 'fork',
         49: 'knife',
         50: 'spoon',
         51: 'bowl',
         52: 'banana',
         53: 'apple',
         54: 'sandwich',
         55: 'orange',
         56: 'broccoli',
         57: 'carrot',
         58: 'hot dog',
         59: 'pizza',
         60: 'donut',
         61: 'cake',
         62: 'chair',
         63: 'couch',
         64: 'potted plant',
         65: 'bed',
         67: 'dining table',
         70: 'toilet',
         72: 'tv',
         73: 'laptop',
         74: 'mouse',
         75: 'remote',
         76: 'keyboard',
         77: 'cell phone',
         78: 'microwave',
         79: 'oven',
         80: 'toaster',
         81: 'sink',
         82: 'refrigerator',
         84: 'book',
         85: 'clock',
         86: 'vase',
         87: 'scissors',
         88: 'teddy bear',
         89: 'hair drier',
         90: 'toothbrush'
}

# Here we will use COLOR_IDS to map each class with a unique RGB color.
R = np.array(np.arange(96, 256, 32))
G = np.roll(R, 1)
B = np.roll(R, 2)

COLOR_IDS = np.array(np.meshgrid(R, G, B)).T.reshape(-1, 3)

## Model Inference using Tensorflow Hub
EfficientDet  = {'EfficientDet D0 512x512'   : 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
                 'EfficientDet D1 640x640'   : 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
                 'EfficientDet D2 768x768'   : 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
                 'EfficientDet D3 896x896'   : 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
                 'EfficientDet D4 1024x1024' : 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
                 'EfficientDet D5 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
                 'EfficientDet D6 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
                 'EfficientDet D7 1536x1536' : 'https://tfhub.dev/tensorflow/efficientdet/d7/1'
                }

def load_model(name='EfficientDet D4 1024x1024'):
    print('loading model: ', name)
    local_path = os.path.join("NN_models", name)
    if os.path.exists(local_path):
        model_url = local_path
        print("Found model locally in url: " + model_url)
        od_model = hub.load(model_url)
    else:
        model_url = EfficientDet[name]
        od_model = hub.load(model_url)

        os.makedirs(local_path)
        tf.saved_model.save(od_model, local_path)

    print('\nmodel loaded!')
    return od_model


## Post-Process and Display Detections
def process_detection(image, results, min_det_thresh=.3):
    # Extract the detection results from the results dictionary.
    scores = results['detection_scores'][0]
    boxes = results['detection_boxes'][0]
    classes = (results['detection_classes'][0]).astype(int)

    # Set a minimum detection threshold to post-process the detection results.
    min_det_thresh = min_det_thresh

    # Get the detections whose scores exceed the minimum detection threshold.
    det_indices = np.where(scores >= min_det_thresh)[0]

    scores_thresh = scores[det_indices]
    boxes_thresh = boxes[det_indices]
    classes_thresh = classes[det_indices]

    print('\nScores over threshold: \n\n', scores_thresh[0:num_dets])
    print('\nDetection Classes: \n\n', classes_thresh[0:num_dets])
    print('\nDetection Boxes: \n\n', boxes_thresh[0:num_dets])

    # Make a copy of the image to annotate.
    img_bbox = image.copy()

    im_height, im_width = image.shape[:2]

    font_scale = .6
    box_thickness = 2

    # Loop over all thresholded detections.
    for box, class_id, score in zip(boxes_thresh, classes_thresh, scores_thresh):

        # Get bounding box normalized coordiantes.
        ymin, xmin, ymax, xmax = box

        class_name = class_index[class_id]

        # Convert normalized bounding box coordinates to pixel coordinates.
        (left, right, top, bottom) = (int(xmin * im_width),
                                      int(xmax * im_width),
                                      int(ymin * im_height),
                                      int(ymax * im_height))

        # Annotate the image with the bounding box.
        color = tuple(COLOR_IDS[class_id % len(COLOR_IDS)].tolist())[::-1]
        img_bbox = cv2.rectangle(img_bbox, (left, top), (right, bottom), color, thickness=box_thickness)

        # -------------------------------------------------------------------
        # Annotate bounding box with detection data (class name and score).
        # -------------------------------------------------------------------

        # Build the text string that contains the class name and score associated with this detection.
        display_txt = '{}: {:.2f}%'.format(class_name, 100 * score)
        ((text_width, text_height), _) = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Handle case when the label is above the image frame.
        if top < text_height:
            shift_down = int(2 * (1.3 * text_height))
        else:
            shift_down = 0

        # Draw a filled rectangle on which the detection results will be displayed.
        img_bbox = cv2.rectangle(img_bbox,
                                 (left - 1, top - box_thickness - int(1.3 * text_height) + shift_down),
                                 (left - 1 + int(1.1 * text_width), top),
                                 color,
                                 thickness=-1)

        # Annotate the filled rectangle with text (class label and score).
        img_bbox = cv2.putText(img_bbox,
                               display_txt,
                               (left + int(.05 * text_width), top - int(0.2 * text_height) + int(shift_down / 2)),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return img_bbox


if __name__=="__main__":

    images = load_images("../imgs", "distorted_1.jpg")
    display_sample_images(images)

    t1 = time.perf_counter_ns()
    od_model = load_model('EfficientDet D7 1536x1536')
    print(f"Model loading time: {1e-9 * (time.perf_counter_ns() - t1)}.")
    ##  Perform Inference
    t1 = time.perf_counter_ns()
    # Call the model. # The model returns the detection results in the form of a dictionary.
    results = od_model(images[0])
    print(f"Detection time: {1e-9 * (time.perf_counter_ns() - t1)}.")

    # Convert the dictionary values to numpy arrays.
    results = {key:value.numpy() for key, value in results.items()}

    # Print the keys from the results dictionary.
    for key in results:
        print(key)

    print('Num Raw Detections: ', (len(results['raw_detection_scores'][0])))
    print('Num Detections:     ', (results['num_detections'][0]).astype(int))

    # Print the Scores, Classes and Bounding Boxes for the detections.
    num_dets = (results['num_detections'][0]).astype(int)

    # print('\nDetection Scores: \n\n', results['detection_scores'][0][0:num_dets])
    # print('\nDetection Classes: \n\n', results['detection_classes'][0][0:num_dets])
    # print('\nDetection Boxes: \n\n', results['detection_boxes'][0][0:num_dets])

    # Remove the batch dimension from the first image.
    image = np.squeeze(images[0])

    # Process the first sample image.
    img_bbox = process_detection(image, results, min_det_thresh=0.2)

    plt.figure(figsize=[15, 10])
    plt.imshow(img_bbox)
    plt.axis('off');
    plt.show()
