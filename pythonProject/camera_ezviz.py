import os
import time

from camera_rtsp import VideoCapture
from pyezviz import EzvizClient, EzvizCamera


class CamEzviz():

    def __init__(self, rtsp_url, ezviz_username, ezviz_password):
        self.cap = VideoCapture(rtsp_url)
        self.client = EzvizClient(ezviz_username, ezviz_password, "eu")

    def scan(self, name: str, path: os.PathLike[any] = "../imgs"):

        img_path_name = os.path.join(path, name)
        try:
            self.client.login()
            camera = EzvizCamera(self.client, "J19619108")
            print(camera.status())
            camera.move_coordinates(0.4 , y_axis=0.0)  # 0.4, 0.5, 0.6
            time.sleep(10)
            self.cap.save_snapshot(img_path_name + "_1.jpg")

            camera.move_coordinates(0.5, y_axis=0.0)  # 0.4, 0.5, 0.6
            time.sleep(10)
            self.cap.save_snapshot(img_path_name + "_2.jpg")

            camera.move_coordinates(0.6, y_axis=0.0)  # 0.4, 0.5, 0.6
            time.sleep(10)
            self.cap.save_snapshot(img_path_name + "_3.jpg")
        except BaseException as exp:
            print(exp)
            return 1
        finally:
            self.client.close_session()
            self.cap.release()


if __name__=="__main__":
    # Load environment variables
    EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
    EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

    RTSP_URL = os.getenv("RTSP_URL")

    cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD)
    cam.scan("test")