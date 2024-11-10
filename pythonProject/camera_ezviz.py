import os
import threading
import time
import typing

import cv2

from camera_calibration import CamIntrinsics
from camera_rtsp import VideoCapture
from pyezviz import EzvizClient, EzvizCamera




class CamEzviz():

    def __init__(self, rtsp_url, ezviz_username, ezviz_password):

        # Ezviz API client
        self.client = EzvizClient(ezviz_username, ezviz_password, "eu")
        # RTSP Video stream
        self.cap = VideoCapture(rtsp_url)
        # Camera parameters
        self.instrinsics = CamIntrinsics("../Ezviz_C6N")

    def scan(self, name: typing.Union[str, None] = None, path: os.PathLike[any] = "../imgs",
             undistort = True):
        """
        Take scan of the scene. Camera rotates and takes multiple pictures.

        :param name: Name of the images to be taken. If None, images are not saved.
        :param path: Path to save the images
        :param undistort: Whether to undistort images
        :return: arr[cv.Mat] images
        """
        out_imgs = []
        if name:
            img_path_name = os.path.join(path, name)
        else:
            img_path_name = None

        try:
            self.client.login()
            camera = EzvizCamera(self.client, "J19619108")
            print(camera.status())

            threads = []
            camera.move_coordinates(0.4 , y_axis=0.0)
            time.sleep(10)
            if undistort:
                self.take_img(img_path_name, "_1.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_1.jpg"))

            camera.move_coordinates(0.5, y_axis=0.0)
            time.sleep(10)
            if undistort:
                self.take_img(img_path_name, "_2.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_2.jpg"))

            for i in range(6):
                camera.move("up")
                time.sleep(2)
            time.sleep(5)
            if undistort:
                self.take_img(img_path_name, "_3.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_3.jpg"))

            for i in range(6):
                camera.move("down")
                time.sleep(2)

            camera.move_coordinates(0.6, y_axis=0.0)
            time.sleep(10)
            if undistort:
                self.take_img(img_path_name, "_4.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_4.jpg"))

            camera.move_coordinates(0.5, y_axis=0.0)

            for thread in threads:
                thread.join()

        except BaseException as exp:
            print(exp)
            return 1

        return out_imgs

    def close(self):
        self.client.close_session()
        self.cap.release()

    def undistort(self, in_images, out_images = None, idx = None):
        """
        Undistorts images. Supply output list and idx for processing in own new thread.

        :param in_images: Input image or a list of images
        :param out_images: Output list for results
        :param idx: Index into output list
        :return: Undistorted image or image list
        """
        if out_images:
            out_images[idx] = self.instrinsics.undistort(in_images)
            return out_images[idx]
        else:
            out_images = []
            if isinstance(in_images, list):
                for img in in_images:
                    out_images.append(self.instrinsics.undistort(img))
                return out_images
            else:
                return self.instrinsics.undistort(in_images)

    def take_img(self, img_name: typing.Union[str, None], suffix: str = "",
                 out_imgs = None, threads = None):
        img = self.cap.read()

        if isinstance(out_imgs, list) and isinstance(threads, list):
            out_imgs.append(None)
            threads.append(threading.Thread(target=self.undistort, args=(img, out_imgs, len(threads))))
            threads[len(threads) - 1].start()

        cv2.imshow("Snapshot", img)
        if img_name:
            cv2.imwrite(img_name + suffix, img)
        cv2.waitKey(1000)
        cv2.destroyWindow("Snapshot")

        return img


if __name__=="__main__":
    # Load environment variables
    EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
    EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

    RTSP_URL = os.getenv("RTSP_URL")

    cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD)
    t1 = time.time_ns()
    images = cam.scan("test", undistort=False)
    rectified_images = cam.undistort(images)
    for i in range(len(rectified_images)):
        cv2.imwrite("rectified_" + str(i+1) + ".jpg", rectified_images[i])

    print("time: " + str(1e-9 * (time.time_ns() - t1)))
    cam.close()