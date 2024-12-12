import cv2
import datetime
import os
from pyezviz import EzvizClient, EzvizCamera
import threading
import time
import tkinter as tk
import typing
from tkinter import DoubleVar

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.camera.camera_rtsp import VideoCapture


class CamEzviz():

    def __init__(self, rtsp_url, ezviz_username, ezviz_password, img_save_dir=None, img_save_name="C6N_IMG",
                 show_video=True):

        # Ezviz API client
        self.client = EzvizClient(ezviz_username, ezviz_password, "eu")
        self.client.login()
        self.camera = EzvizCamera(self.client, "J19619108")
        self.camera.move_coordinates(x_axis=0.5, y_axis=0.0)
        print(self.camera.status())

        # RTSP Video stream
        self.cap = VideoCapture(rtsp_url, show_video)
        # Camera parameters
        self.instrinsics = CamIntrinsics(os.path.join(os.path.dirname(__file__), "Ezviz_C6N"))

        if img_save_dir is None:
            # Get the absolute path of the current script
            script_path = os.path.abspath(__file__)

            # Get the directory containing the script
            script_dir = os.path.dirname(script_path)

            img_save_dir = os.path.join(script_dir, "imgs")
            # Check if the folder exists
            if not os.path.exists(img_save_dir):
                # Create the folder
                os.makedirs(img_save_dir)

        self.img_save_dir = img_save_dir
        self.img_save_name = img_save_name

    def control(self):
        # Create a window
        root = tk.Tk()

        # Create a slider for X-Axis
        var_x = DoubleVar()
        var_x.set(0.5)
        def slider_x_on_change(val):
            var_x.set(val)
            self.camera.move_coordinates(x_axis=var_x.get(), y_axis=0)  # y_axis move does not work due to EZVIZ API

        x_res = 0.02
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
                img = self.take_img(f"{self.img_save_dir}/{self.img_save_name}_{datetime_str}")
                rect_img = self.undistort(img)
                cv2.imwrite(f"{self.img_save_dir}/{self.img_save_name}_{datetime_str}_rect.jpg", rect_img)

        root.bind("<KeyPress>", on_key_press)

        def exit(e):
            root.destroy()

        root.bind("<Escape>", exit)
        root.mainloop()


    def scan(self, name: typing.Union[str, None] = None, path: os.PathLike[any] = None,
             undistort = True):
        """
        Take scan of the scene. Camera rotates and takes multiple pictures.

        :param name: Name of the images to be taken. If None, images are not saved.
        :param path: Path to save the images
        :param undistort: Whether to undistort images
        :return: arr[cv.Mat] images
        """
        if path is None:
            path = self.img_save_dir

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

    def is_opened(self):
        return self.cap.is_opened()


    def close(self):
        self.client.close_session()
        self.cap.release()

    def undistort(self, in_images, #: typing.Union[cv2.typing.MatLike| list[cv2.typing.MatLike]],
                  out_images: list = None, idx: int = None):
        """
        Undistort one or more images.

        :param in_images: Input image or a list of images
        :param out_images: Output list for results.
        :param idx: Index into output list
        :return: Undistorted image or image list
        """

        if out_images is None:
            out_images = []

        # Undistort List of images
        if isinstance(in_images, list):
            return self.instrinsics.undistort(in_images)
        # Undistort a single image
        else:
            if ((out_images is not None) and
                    (idx is not None) and
                    (idx < len(out_images))):
                # Note: This is intended for threaded undistortion
                # ToDo: Debug and Fix
                out_images[idx] = self.instrinsics.undistort_img(in_images)
                return out_images[idx]
            else:
                return self.instrinsics.undistort_img(in_images)

    def take_img(self, img_name: typing.Union[str, None] = None, suffix: str = "",
                 out_imgs = None, threads = None):
        """Take an image from VideoCapture.

        ToDo: Debug and correct threading implementation. As it does not work 100% at the moment.
        :param img_name: Image name. If specified, image is saved to disk.
        :param suffix: Optional img_name suffix
        :param out_imgs: Output images array. If specified, rectification is performed in a new thread.
        :param threads:  List with pointers to rectification threads
        """

        if suffix == "":
            suffix = ".jpg"

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

    IMG_DIR = os.getenv("IMG_DIR")

    RTSP_URL = os.getenv("RTSP_URL")

    cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir=IMG_DIR)

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