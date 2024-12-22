import os
import re

import cv2
import threading

from pythonProject.camera.fps import FPS


# bufferless VideoCapture
class VideoCapture:
    """VideoCapture device thread safe implementation.

    Usage:
        - read() - read the latest camera frame
        - press 's' to save latest image
        - press 'ESC' to quit capturing thread

    """
    def __init__(self, name, show_video=False):

        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        self.img = None
        self.show_video = show_video
        self.t = threading.Thread(target=self._reader, args=(show_video,))
        # self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self, show_video):
        snapshots_no = 0
        # Find how many snapshots have been taken in order to adjust name numbers
        for root, dirs, files in os.walk("../../imgs"):
            for name in files:
                if re.search(r"^snapshot_[0-9]+\.jpg$", name):
                    snapshots_no += 1

        fps = FPS().start()

        while True:
            with self.lock:
                ret, self.img = self.cap.read()
                if not ret:
                    break

            if show_video:
                fps.update()

                fps.add_to_image(self.img)

                cv2.imshow("VideoCapture", cv2.resize(self.img, (960, 540)))

            key_press = cv2.waitKey(1)
            if  key_press == ord('s'):  # Save a snapshot when 's' is pressed
                snapshots_no += 1
                snapshot_name = "snapshot_" + str(snapshots_no) + ".jpg"
                self.save_snapshot(os.path.join("../imgs", snapshot_name))
            elif key_press == 27:       # ESC for quit
                break

        self.release()

    def is_opened(self):
        return self.cap.isOpened()

    # retrieve latest frame
    def read(self):
        with self.lock:
            if self.img is not None:
                return self.img.copy()
            else:
                return None

    def release(self):
        with self.lock:
            self.cap.release()
            # Test if VideoCapture window is open
            if self.show_video:
                cv2.destroyWindow("VideoCapture")

    def save_snapshot(self, img_name):
        with self.lock:
            cv2.imshow("Snapshot", self.img)
            cv2.imwrite(img_name, self.img)
        cv2.waitKey(1000)
        cv2.destroyWindow("Snapshot")


if __name__ == '__main__':
    RTSP_URL = os.getenv("RTSP_URL")
    cap = VideoCapture(RTSP_URL)
    print(cap.__doc__)
