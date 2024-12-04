import cv2
import numpy as np


class MouseSketcher:
    """Class serving for arbitrary shape selection using mouse drag."""

    def __init__(self, windowname, dests):
        self.polygon_points = []
        self.drawing = False

        self.windowname = windowname
        self.dests = dests
        self.mask = np.zeros(dests.shape[:2], dtype=np.uint8)

        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests)

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback function. Identify occurred event and perform defined functionality.

        :param event: The type of mouse event.
                - cv2.EVENT_LBUTTONDOWN: Start recording mouse position when L button was pressed.
                - cv2.EVENT_MOUSEMOVE: Save mouse position
                - cv2.EVENT_LBUTTONUP: Stop recording mouse move
        :param x, y: Mouse position
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.polygon_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.polygon_points.append((x, y))
            if len(self.polygon_points) > 1:
                cv2.polylines(self.dests, [np.array(self.polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.imshow(self.windowname, self.dests)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.polygon_points) > 2:
                cv2.fillPoly(self.mask, [np.array(self.polygon_points)], color=255)

            self.polygon_points.clear()



def create_mask(image):
    """Draw and save bitmap mask of arbitrary shape.

    Draw a mask on top of input image using MouseSketcher.
    Handling:
        - 'L mouse down' : Start and record mouse position
        - 'L mouse up'   : Stop recording mouse position
        - 'Esc'          : Exit sketch
        - 'r'            : Mask the image
        - 'Space'        : Erase/Reset drawn mask

    :param image: Background image on which to draw mask.
    :return np.ndarray: Drawn 1 channel mask.
    """

    # Create an image for sketching the mask
    image_mark = image.copy()
    sketch = MouseSketcher('Image', image_mark)

    # Sketch a mask
    while True:
        ch = cv2.waitKey()
        if ch == 27:  # ESC - exit
            break
        if ch == ord('r'):  # r - mask the image
            break
        if ch == ord(' '):  # SPACE - reset the inpainting mask
            image_mark[:] = image
            sketch.show()

    mask = sketch.mask
    cv2.imshow("mask", mask)
    # cv2.imwrite("base_rect_mask.jpg", mask)

    return mask