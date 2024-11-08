# Parking Spot Detector
This project aim is to develop an algorithm to detect available parking spots and 
inform a driver via SMS, Call or Website, where he/she can park, 
when returning home. 

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
   ![panoramas](./imgs/distorted_pano.jpg)
1) Create panoramatic picture
   * Take several pictures while rotating the camera
   * Stitch them together using OpenCV Stitcher algorithm in order to get panoramatic picture over the parking lot.