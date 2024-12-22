# import the necessary packages
import datetime

import cv2
import numpy as np


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def add_to_image(self, img: np.ndarray):
        # Display FPS on the video frame
        fps_text = f"FPS: {self.fps():.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)  # Černý text
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        position = (10, 30)
        background_color = (255, 255, 255)

        # Vykreslete obdélník jako pozadí
        cv2.rectangle(img,
                      (position[0] - 10, position[1] - text_height - 10),
                      (position[0] + text_width + 10, position[1] + 10),
                      background_color,
                      -1)
        cv2.putText(img, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

        return img

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._end = datetime.datetime.now()
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()