from PIL import Image, ImageDraw
import numpy as np
import os 

# Function to create a mask from polygon
def create_mask_from_polygon(image_size, polygon, i):
    mask = Image.new('L', image_size, 0)  # Create a new black image for the mask
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)  # Draw the polygon on the mask
    return mask

def get_bounding_box(image_np):
    coords = np.argwhere(image_np != 0)
    print(coords)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return (x0, y0, x1, y1)

# Process each line in the label file
def crop_images(image_path, label_path, dest):
    # image_path = 'test/images/1_20211204_111234_jpg.rf.dd862269207c26e54f6183784de2483a.jpg'
    image = Image.open(image_path)
    image_id = image_path.split('.')[2]
    # label_path = 'test/labels/1_20211204_111234_jpg.rf.dd862269207c26e54f6183784de2483a.txt'
    
    # Get dimension image
    x, y = image.size
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        # Group coordinates into pairs
        polygon_coords = [(coords[j] * x, coords[j + 1]* y) for j in range(0, len(coords), 2)]

        # Create a mask for the polygon
        mask = create_mask_from_polygon(image.size, polygon_coords, i)

        # Convert mask to numpy array
        mask_np = np.array(mask)
        # Crop the image using the mask
        image_np = np.array(image)
        white_background = np.ones_like(image_np) * 255

        # crop image with white background
        masked_image_np = np.where(mask_np[:, :, None] == 1, image_np, white_background)
        crop_boundingbox = get_bounding_box(mask_np)
        masked_image = Image.fromarray(masked_image_np)
        masked_image = masked_image.crop(crop_boundingbox)

        # Save the cropped image
        cropped_image_path = f'{dest}/{class_id}_{image_id}_{i}.png'
        if not os.path.exists(dest):
            os.makedirs(dest)

        masked_image.save(cropped_image_path)
        print(f'Cropped image saved to {cropped_image_path}')