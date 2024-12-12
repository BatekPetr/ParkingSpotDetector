# Parking Spot Detector
This project aim is to develop an algorithm to detect available parking spots and 
inform a driver via SMS, Call or Website, where he/she can park, 
when returning home. Parking spaces are available along the street and therefore 
camera has to be rotated and more images have to be taken in order to cover the 
whole area.

## Used HW and SW
* EZVIZ Camera:
  * https://www.ezviz.com/product/c6n/9046
  * note: in order to be able to read video stream via RTSP protocol an older firmware has to be used.
  Downgrade the camera firmware following the tutorial from: https://ipcamtalk.com/threads/ezviz-disables-rtps-for-its-ip-cameras-here-is-the-fix.69927/
* PyEzviz:
  * https://pypi.org/project/pyezviz/
  * https://github.com/baqs/pyEzviz/
  * Unofficial Python package for controlling Ezviz camera implementing python-Ezviz API.
* OpenCV
* Python

## Image processing pipeline
0) Find Camera intrinsic parameters for image rectification.
   * When pictures are not rectified, final panoramas are distorted and does not look pretty.
   ![distorted-panorama](./imgs/distorted_pano.jpg)
   * Stitching of undistorted images looks better
   ![undistorted_panorama](./imgs/undistorted_pano.jpg)
1) Create panoramatic picture
   * Take several pictures while rotating the camera.
   * Undistort images.
   * Original idea was to use OpenCV Sticher class to perform stitching. 
   However, it turned out to be not suitable because Stitcher algorithm does not output homography transformations,
   which are needed to align a new panorama with parking slot template, where parking slots are defined.
   * Own stitching algorithm was developed in [MyStitching](./pythonProject/stitching/my_stitching.py) module. It offers
   following functionality:
     * Recursively stitches list of images of arbitrary length.
     ![template_pano.jpg](./imgs/template_pano.jpg) *Fig: Template Panorama
     * Outputs final homogeneous transformations of individual images to the resulting panorama.
     * Buffers and transforms image keypoints to avoid unnecessary multiple computations.
     * Image warping (cylindrical and spherical) was tested to avoid Edge stretching, where objects near the edges 
     of the panorama appear larger than those in the center. 
     ![Pano Spherical Warp](./imgs/template_pano_spherical.jpg)
     *Fig: Panorama with Spherical Warp
     
     As can be seen on [Pano Spherical Warp](./imgs/template_pano_spherical.jpg) warping corrects Edge stretching 
     however images are not properly stitched together. Another disadvantage of warping is that it does not 
     preserve straight lines. Having straight lines is beneficial in rectangular parking slots selection. 
     Therefore, no warping prior to stitching was performed. Warping can be applied to final panorama to slightly 
     improve the visual effect.
   * SIFT and ORB feature detection and matching was implemented (using OpenCV functions). SIFT proved to be more 
   accurate and robust in cost of longer processing time. The cost is acceptable by the usecase, so SIFT is used primarily.
   * ToDo: Image blending was not yet implemented as it is not necessary for the usecase.
2) Use Neural Network to detect parked vehicles.
   1) The first "naive" approach was to use existing pre-trained NN models. 
      * TensorFlow implementation of model [EfficientDet](https://www.kaggle.com/models/tensorflow/efficientdet/tensorFlow2/d7) 
      was initially tested. However, the model did not prove to be suitable as can be seen on the following image.
      ![TF_detection](./imgs/TF_detection.jpg)
      * The second test was with YOLO models. The detection was not much better.
   2) This experience lead the project to the task of training a custom NN. This task required a small research to be performed 
   in order to choose suitable tools and NNs architecture.
3) Research about NN image detections
   * Before choosing one of the options research about the topic of car detections was performed.
   * These insights were acknowledged:
     * The NNs trained on general datasets like [COCO](https://cocodataset.org/#home) are too general for the project's specific use-case.
     * [Roboflow](https://roboflow.com/) platform was found. It contains many NN models and image datasets from various use-cases.
     * [UltralyticsHub](https://www.ultralytics.com/hub) platform found. It offers seamless transfer of image datasets from Roboflow
     and is very helpful for training models. Google Colab code generation can be used for training models.
     * [Ultralytics](https://docs.ultralytics.com/) company offers implementations of YOLO NN models. 
   Implementation of YOLOv11 is in the [Python package Ultralytics](https://pypi.org/project/ultralytics/), 
   whereas older version such as YOLOv8 can be found on [Github](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md).
     * [Google Colab](https://colab.research.google.com/) was used for computing extensive tasks as it offers a use of GPU resources.
     * [Google Kaggle](https://www.kaggle.com) similar platform as Colab for use of GPUs for computing.
     * There are to types of NNs for object detection:
       * Multipass NNs - detection and labelling happens in 2 or more passes through NN. Due to this property, detection is usually longer but can be more precise.
       * Singlepass NNs or Single Shot Detectors (SSD) - detection and labelling happens during a single pass through NN. 
       This architecture results in faster detection for the cost of worse accuracy.
     * [Autodistill](https://docs.autodistill.com/) library can be used for auto-labelling initial images.
4) Dataset creation
   * All images where taken from the Ezviz camera
   1) For labelling initial 100 images, Autodistill library was used. GroundedDINO model was used for detections and labelling. 
   It was help-full, but detections required significant manual corrections 
   and adding box-labels for a lot of missed detections.
   2) After the first image set creation a custom model of YOLOv11x architecture was trained. This model performs much better detection than general models.
   3) Another 500 images were auto-labelled using the first custom YOLO model version.