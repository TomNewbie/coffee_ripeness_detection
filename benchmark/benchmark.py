from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

# Load a model
# model = YOLO("yolov8n.pt")  # load an official model
yaml_file_path='/home/topaci/Documents/thesis/sandbox/dataset/ivyqo_augment_dataset/data.yaml'

# model = YOLO("yolov8m-e50.pt")  # load a custom trained model
benchmark(model="yolov8m-e50.pt", data=yaml_file_path, imgsz=800, half=False, device='cpu')

# Export the model
# model.export(format="onnx")
# model.predict