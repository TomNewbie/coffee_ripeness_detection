from ultralytics import YOLO, solutions
from PIL import Image, ImageDraw, ImageFont

# Load a model
model = YOLO("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/evaluate/yolov8.pt") 

results = model("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/dataset/ivyqo_augment_dataset/test/images")

def drawText(img_path, localtion, key, value):
    img = Image.open(img_path)
    font = ImageFont.load_default(size=25)
    img.text(localtion, f"{key}: {value}", fill="white", font=font)


for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    with open('./result.txt', 'w') as file:
        file.write(str(result))
    # for key in result
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
print(result)
