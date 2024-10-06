from ultralytics import YOLO, RTDETR
from PIL import Image, ImageDraw, ImageFont

# Load a model
# model = YOLO("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/evaluate/yolov8.pt") 
model = RTDETR("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/evaluate/DETR_Batch5_best.pt") 

# results = model("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/dataset/ivyqo_augment_dataset/test/images/20240324_113346_jpg.rf.7459de60040f94ba83e9598f2016c39f.jpg")

def drawText(img_path, localtion, key, value):
    img = Image.open(img_path)
    font = ImageFont.load_default(size=25)
    img.text(localtion, f"{key}: {value}", fill="white", font=font)


# for result in results:
#     # boxes = result.boxes  # Boxes object for bounding box outputs
#     # masks = result.masks  # Masks object for segmentation masks outputs
#     # keypoints = result.keypoints  # Keypoints object for pose outputs
#     # probs = result.probs  # Probs object for classification outputs
#     # obb = result.obb  # Oriented boxes object for OBB outputs
#     with open('./result.txt', 'w') as file:
#         file.write(str(result))
#     # for key in result
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk
#     print(result)
result = model.predict("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/dataset/ivyqo_augment_dataset/test/images", save= True, save_txt=True)
print(result)
