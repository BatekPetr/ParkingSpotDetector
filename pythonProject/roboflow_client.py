import roboflow

if __name__=="__main__":
    rf = roboflow.Roboflow(api_key="SApF3gugLbabAlvBadWM")
    project = rf.workspace().project("cardetector-kkdtp")

    #can specify weights_filename, default is "weights/best.pt"
    version = project.version(5)
    #version.deploy("model-type", "path/to/training/results/", "weights_filename")

    #example1 - directory path is "training1/model1.pt" for yolov8 model
    version.deploy("yolov11", "NN_models", "YOLOv11x_MyDataset_imgsz1024.pt")