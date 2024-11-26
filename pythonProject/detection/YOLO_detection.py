import cv2
import os
from ultralytics import YOLO

from .detection import load_images


class YOLODetector:

    def __init__(self, path_to_yolo_weights: str):
        self.model = YOLO(path_to_yolo_weights)

    def detect_in_images(self, file_pattern: str, folder: str="../imgs"):
        images = load_images("../imgs", "distorted_*.jpg")

        for image in images:
            results = self.model(image)

            for result in results:
                # get the classes names
                classes_names = result.names

                # iterate over each box
                for box in result.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.1:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = classes_names[cls]

                        # get the respective colour
                        colour = self.get_colors(cls)

                        # draw the rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

                        # put the class name and confidence on the image
                        cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                # show the image
            cv2.imshow('frame', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_in_video(self, cam):

        image = cam.take_img()

        while image is not None:
            results = self.model(image)

            for result in results:
                # get the classes names
                classes_names = result.names

                # iterate over each box
                for box in result.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.1:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = classes_names[cls]

                        # get the respective colour
                        colour = self.get_colors(cls)

                        # draw the rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

                        # put the class name and confidence on the image
                        cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                # show the image
            cv2.imshow('frame', cv2.resize(image, (960, 540)))
            image = cam.take_img()

        cv2.destroyWindow('frame')


    # Function to get class colors
    def get_colors(self, cls_num):
        # cls_num = len(self.model.names)
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] *
        (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)


if __name__=="__main__":

    model = YOLODetector(os.path.join("./NN_models", "YOLOv11n_MyDataset_imgsz640.pt"))
    # model.detect_in_images("distorted_3.jpg", folder="../imgs")

    # Load environment variables
    EZVIZ_USERNAME = os.getenv("EZVIZ_USERNAME")
    EZVIZ_PASSWORD = os.getenv("EZVIZ_PASSWORD")

    RTSP_URL = os.getenv("RTSP_URL")
    from pythonProject.camera.camera_ezviz import CamEzviz
    cam = CamEzviz(RTSP_URL, EZVIZ_USERNAME, EZVIZ_PASSWORD, img_save_dir="../../imgs/parking_dataset", show_video=False)

    import threading
    t = threading.Thread(target=model.detect_in_video, args=(cam,))
    t.daemon = True
    t.start()

    cam.control()
    cam.close()