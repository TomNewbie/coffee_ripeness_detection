from ultralytics import RTDETR, solutions
# from PIL import Image, ImageDraw, ImageFont

# Load a model
model = RTDETR("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/evaluate/DETR_Batch5_best.pt") 

results = model("/media/topaci/DATA/Thophan/CS2020/thesis/sandbox/dataset/ivyqo_augment_dataset/test/images")

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
# print(result)
