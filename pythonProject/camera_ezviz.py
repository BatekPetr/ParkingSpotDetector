import datetime
import os
import re
import threading
import time
import tkinter as tk
import typing
from tkinter import DoubleVar

import cv2

from camera_calibration import CamIntrinsics
from camera_rtsp import VideoCapture
from pyezviz import EzvizClient, EzvizCamera


class CamEzviz():

    def __init__(self, rtsp_url, ezviz_username, ezviz_password, img_save_dir="../imgs", img_save_name="C6N_IMG"):

        # Ezviz API client
        self.client = EzvizClient(ezviz_username, ezviz_password, "eu")
        self.client.login()
        self.camera = EzvizCamera(self.client, "J19619108")
        self.camera.move_coordinates(x_axis=0.5, y_axis=0.0)
        print(self.camera.status())

        # RTSP Video stream
        self.cap = VideoCapture(rtsp_url)
        # Camera parameters
        self.instrinsics = CamIntrinsics("../Ezviz_C6N")

        self.img_save_dir = img_save_dir
        self.img_save_name = img_save_name
        # try:
        #     saved_images = os.listdir(img_save_dir)
        #     saved_images.sort()
        #
        #     re_match = re.search(rf"{img_save_name}_([0-9])*", saved_images[-1])
        #     last_img_no = re_match.group(1)
        #     if last_img_no:
        #         self.img_no = int(last_img_no) + 1
        #     else:
        #         self.img_no = 1
        # except FileNotFoundError:
        #     self.img_no = 1

    def control(self):
        # Create a window
        root = tk.Tk()

        # Create a slider for X-Axis
        var_x = DoubleVar()
        var_x.set(0.5)
        def slider_x_on_change(val):
            var_x.set(val)
            self.camera.move_coordinates(x_axis=var_x.get(), y_axis=0)  # y_axis move does not work due to EZVIZ API

        x_res = 0.05
        x = tk.Scale(root, from_=0.3, to=0.7, resolution=x_res, variable=var_x,
                     orient=tk.HORIZONTAL, command=slider_x_on_change)
        x.pack()

        # Create a keyboard listener
        def on_key_press(event):
            if event.keysym == 'Up':
                print("Up arrow pressed")
                self.camera.move("up")
            elif event.keysym == 'Down':
                print("Down arrow pressed")
                self.camera.move("down")
            elif event.keysym == 'Left':
                print("Left arrow pressed")
                slider_x_on_change(var_x.get() - x_res)
            elif event.keysym == 'Right':
                print("Right arrow pressed")
                slider_x_on_change(var_x.get() + x_res)
            elif event.keysym == 's':
                print("'s' for Save pressed")
                datetime_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")
                img = self.take_img(f"{self.img_save_dir}/{self.img_save_name}_{datetime_str}.jpg")
                rect_img = self.undistort(img)
                cv2.imwrite(f"{self.img_save_dir}/{self.img_save_name}_{datetime_str}_rect.jpg", rect_img)
                # self.img_no += 1
            # elif event.keysym == 'ESC':

        root.bind("<KeyPress>", on_key_press)

        def exit(e):
            root.destroy()

        root.bind("<Escape>", exit)
        root.mainloop()


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
            threads = []
            self.camera.move_coordinates(0.4 , y_axis=0.0)
            time.sleep(10)
            if undistort:
                self.take_img(img_path_name, "_1.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_1.jpg"))

            self.camera.move_coordinates(0.5, y_axis=0.0)
            time.sleep(10)
            if undistort:
                self.take_img(img_path_name, "_2.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_2.jpg"))

            for i in range(6):
                self.camera.move("up")
                time.sleep(2)
            time.sleep(5)
            if undistort:
                self.take_img(img_path_name, "_3.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_3.jpg"))

            for i in range(6):
                self.camera.move("down")
                time.sleep(2)

            self.camera.move_coordinates(0.6, y_axis=0.0)
            time.sleep(10)
            if undistort:
                self.take_img(img_path_name, "_4.jpg", out_imgs, threads)
            else:
                out_imgs.append(self.take_img(img_path_name, "_4.jpg"))

            self.camera.move_coordinates(0.5, y_axis=0.0)

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

        #cv2.imshow("Snapshot", img)
        if img_name:
            cv2.imwrite(img_name + suffix, img)
        #cv2.waitKey(1000)
        #cv2.destroyWindow("Snapshot")

        return img


if __name__=="__main__":
    # Load environment variables
    EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
    EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

    RTSP_URL = os.getenv("RTSP_URL")

    cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir="../imgs/parking_dataset")

    ## Perform camera scan, rectify and save images
    # t1 = time.time_ns()
    # images = cam.scan("test", undistort=False)
    # rectified_images = cam.undistort(images)
    # for i in range(len(rectified_images)):
    #     cv2.imwrite("rectified_" + str(i+1) + ".jpg", rectified_images[i])
    #
    # print("time: " + str(1e-9 * (time.time_ns() - t1)))

    ## Control camera using keyboard
    cam.control()

    cam.close()