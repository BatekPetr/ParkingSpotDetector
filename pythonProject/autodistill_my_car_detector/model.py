import time

import os
from dataclasses import dataclass

import torch
from ultralytics import YOLO
import supervision as sv
import autodistill
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class MyCarDetector(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, model_name:str, ontology: CaptionOntology=None):
        self.ontology = ontology
        self._model = YOLO(model_name)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = autodistill.helpers.load_image(input, return_format="cv2")
        result = self._model(image)[0]

        detections = sv.Detections.from_ultralytics(result)

        return detections


if __name__ == "__main__":
    model = MyCarDetector("YOLOv11x_MyDataset_imgsz1024.pt")
    detections = model.predict("../../../imgs/distorted_1.jpg")

    time.sleep(5)