from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("yolov8m-e50.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")