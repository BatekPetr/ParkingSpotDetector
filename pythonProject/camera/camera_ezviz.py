import queue

import cv2
import datetime
import os

import numpy as np
from lxml.etree import SerialisationError
from pyezviz import EzvizClient, EzvizCamera
import multiprocessing
import threading
import time
import tkinter as tk
import typing
from tkinter import DoubleVar

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.camera.camera_rtsp import VideoCapture
from pythonProject.stitching.my_stitching import MyStitcher


class DeviceError(Exception):
    """Custom exception for device-related errors."""
    pass


class CamEzviz:

    def __init__(self, rtsp_url, ezviz_username, ezviz_password, img_save_dir=None, img_save_name="C6N_IMG",
                 show_video=True, use_multiprocessing=False):

        # Ezviz API client
        self.client = EzvizClient(ezviz_username, ezviz_password, "eu")
        self.client.login()
        self.camera = EzvizCamera(self.client, "J19619108")
        self.camera.move_coordinates(x_axis=0.5, y_axis=0.0)
        print(self.camera.status())


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

        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            self.queue = multiprocessing.Queue()
            self.command_pipe, process_pipe = multiprocessing.Pipe()
            self.p = multiprocessing.Process(target=self.video_capture_process, args=(rtsp_url, show_video, process_pipe))
            self.p.start()
        else:
            # RTSP Video stream
            self.cap = VideoCapture(rtsp_url, show_video)

    def video_capture_process(self, rtsp_url, show_video, process_pipe):
        # RTSP Video stream
        cap = VideoCapture(rtsp_url, show_video)

        # Wait until VideoCapture is opened
        while not cap.is_opened():
            time.sleep(0.1)

        # Manage Video Capture process
        while True:
            if process_pipe.poll():
                command = process_pipe.recv()
                if "read" == command:
                    img = cap.read()
                    if img is None:
                        self.queue.put(ValueError("Image is empty or could not be loaded. Camera is not ready."))
                    else:
                        # Serialize the image using cv2.imencode
                        success, encoded_image = cv2.imencode('.jpg', img)
                        if success:
                            self.queue.put(encoded_image.tobytes())  # Put the byte array into the queue
                        else:
                            raise SerialisationError
                elif "stop" == command:
                    cap.release()
                    break

        process_pipe.close()

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

    def undistortion_thread(self, q_in: queue.Queue, q_out: queue.Queue):
   # ToDo: move undistortion_thread to CamIntrinsics
        while True:
            img = q_in.get()
            if img is None:
                break
            rect_img = self.undistort(img)
            q_out.put(rect_img)

    def is_opened(self):
        if self.use_multiprocessing:
            return self.p.is_alive()
        else:
            return self.cap.is_opened()


    def close(self):
        self.client.close_session()

        if self.use_multiprocessing:
            self.command_pipe.send("stop")
            self.command_pipe.close()
            self.p.join()
        else:
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

        img = None
        start_t = time.perf_counter()
        while img is None:
            if self.use_multiprocessing:
                if self.command_pipe.closed:
                    break

                self.command_pipe.send("read")

                # Get the serialized image from the queue
                from_video_process = self.queue.get()
                if isinstance(from_video_process, bytes):
                    encoded_image = from_video_process

                    # Deserialize the image using cv2.imdecode
                    nparr = np.frombuffer(encoded_image, np.uint8)  # Convert bytes back to NumPy array
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image
            else:
                if self.cap.is_opened():
                    img = self.cap.read()
                else:
                    break

            # Raise device error if cam image is unavailable for 10 s
            if time.perf_counter() - start_t > 10:
                raise DeviceError

        if isinstance(out_imgs, list) and isinstance(threads, list):
            out_imgs.append(None)
            threads.append(threading.Thread(target=self.undistort, args=(img, out_imgs, len(threads))))
            threads[len(threads) - 1].start()

        if img_name:
            cv2.imwrite(img_name + suffix, img)

        return img


if __name__=="__main__":
    # Load environment variables
    EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
    EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

    IMG_DIR = os.getenv("IMG_DIR")

    RTSP_URL = os.getenv("RTSP_URL")

    # No significant improvement when multiprocessing used here
    cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir=IMG_DIR, use_multiprocessing=True)

    # Control camera using keyboard
    cam.control()

    cam.close()