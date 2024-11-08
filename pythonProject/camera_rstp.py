import os

import cv2
from pyezviz import EzvizClient, EzvizCamera
import sys
import threading
import time

# Load environment variables
EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

RTSP_URL = os.getenv("RTSP_URL")

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        self.img = None
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret, self.img = self.cap.read()
            if not ret:
                break
            else:
                cv2.imshow("video output", cv2.resize(self.img, (960, 540)))
                cv2.waitKey(1)

    # retrieve latest frame
    def read(self):
        with self.lock:
            return self.img

    def release(self):
        with self.lock:
            self.cap.release()

    def save_snapshot(self, img_name):
        with self.lock:
            cv2.imshow('snapshot', self.img)
            cv2.imwrite(img_name, self.img)
        cv2.waitKey(1000)
        cv2.destroyWindow('snapshot')

def scan(name: str, path: os.PathLike[any] = "../imgs"):
    cap = VideoCapture(RTSP_URL)
    client = EzvizClient(EZVIZ_USERNAME, EZVIZ_PASSWORD, "eu")
    img_path_name = os.path.join(path, name)
    try:
        client.login()
        camera = EzvizCamera(client, "J19619108")
        print(camera.status())
        camera.move_coordinates(0.4 , y_axis=0.0)  # 0.4, 0.5, 0.6
        time.sleep(10)
        cap.save_snapshot(img_path_name + "_1.jpg")

        camera.move_coordinates(0.5, y_axis=0.0)  # 0.4, 0.5, 0.6
        time.sleep(10)
        cap.save_snapshot(img_path_name + "_2.jpg")

        camera.move_coordinates(0.6, y_axis=0.0)  # 0.4, 0.5, 0.6
        time.sleep(10)
        cap.save_snapshot(img_path_name + "_3.jpg")
    except BaseException as exp:
        print(exp)
        return 1
    finally:
        client.close_session()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(scan("distorted"))
