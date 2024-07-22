from PIL import Image, ImageDraw, ImageFont, ImageColor
import os
from image_utils import save_predict_and_groundtruth
from file_utils import list_all_image_file
import json

# image_path = 'dataset/ivyqo_regular_dataset/test/images/20240324_105525_jpg.rf.34e2ea13e6d402ac79f8bd9032fd5c95.jpg'

folder = './draw-box'

src_folder ='/home/topaci/Documents/thesis/sandbox/dataset/ivyqo_augment_dataset/test/images'
os.makedirs(folder, exist_ok=True)


# Path to your JSON file
file_path = './filter.json'

# Read the JSON file
with open(file_path, 'r') as file:
    image_data = json.load(file)

# Now 'data' is a Python dictionary
# print(data)
switcher = {
    0: 'overripe',
    1: 'ripe',
    2: 'semiripe',
    3: 'unripe'
}
print(ImageColor.getrgb)
for data in image_data:
    # if data['score'] < 0.5:
    #     continue
    bbox = data['bbox']
    image_name = data['image_id']
    output_image_path = f'{folder}/{image_name}.jpg'
    origin_image_path = f'{src_folder}/{image_name}.jpg'    

    if os.path.exists(output_image_path):
        image = Image.open(output_image_path)  
    else:
        image = Image.open(origin_image_path)
    
    draw = ImageDraw.Draw(image)


    # Convert bounding box to integer values
    x, y, w, h = map(int, bbox)
    score = data['score']
    confidence_score_text = f"{score:.2f}"
    font = ImageFont.load_default(size=25)

    confidence_score_text_location = (x, y + 10)
    draw.text(confidence_score_text_location, confidence_score_text, fill="white", font=font)

    class_text_location = (x, y + h - 30)
    class_test = f"{switcher[int(data['category_id'])]}"
    draw.text(class_text_location, class_test, fill="white", font=font)

    # Draw the bounding box on the image
    draw.rectangle([x, y, x + w, y + h], outline="blue", width=4)

    # Save the output image

    # print(f"Bounding box drawn and saved to {output_image_path}")
    image.save(output_image_path)

image_files = list_all_image_file(folder)
for image_file in image_files:
    output_image_path = f'{folder}/{image_file}'
    origin_image_path = f'{src_folder}/{image_file}'        
    save_predict_and_groundtruth(output_image_path, origin_image_path, './compare')