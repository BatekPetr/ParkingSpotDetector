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
   * Stitch them together using OpenCV Stitcher algorithm in order to get panoramatic picture over the parking lot.
2) Use Neural Network to detect parked vehicles.
   1) The first "naive" approach was to use existing pre-trained NN models. 
      * TensorFlow implementation of model [EfficientDet](https://www.kaggle.com/models/tensorflow/efficientdet/tensorFlow2/d7) 
      was initially tested. However, the model did not prove to be suitable as can be seen on the following image.
      ![TF_detection](./imgs/TF_detection.jpg)
      * The second test was with YOLO models. The detection was not much better.
   2) This experience lead the project to one of the following options:
      a) Train an own NN.
      b) Use transfer learning to improve precision of detection by re-training existing NN for this individual use-case.
3) Research about NN image detections
   * Before choosing one of the options a research about the topic of car detections was performed.
   * These insights were acknowledged:
     * The NNs trained on general datasets like [COCO](https://cocodataset.org/#home) are too general for the project's specific use-case.
     * [Roboflow](https://roboflow.com/) platform was found. It contains many NN models and image datasets from various usecases.