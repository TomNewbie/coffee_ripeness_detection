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
# def get_crop_images_np(image_path, label_path, dest):
#     # image_path = 'test/images/1_20211204_111234_jpg.rf.dd862269207c26e54f6183784de2483a.jpg'
#     # label_path = 'test/labels/1_20211204_111234_jpg.rf.dd862269207c26e54f6183784de2483a.txt'
#     image = Image.open(image_path)
#     image_id = image_path.split('.')[2]

#     # Get dimension image
#     x, y = image.size
#     with open(label_path, 'r') as file:
#         lines = file.readlines()

#     for i, line in enumerate(lines):
#         parts = line.strip().split()
#         class_id = parts[0]
#         coords = list(map(float, parts[1:]))

#         # Group coordinates into pairs
#         polygon_coords = [(coords[j] * x, coords[j + 1]* y) for j in range(0, len(coords), 2)]

#         # Create a mask for the polygon
#         mask = create_mask_from_polygon(image.size, polygon_coords, i)

#         # Convert mask to numpy array
#         mask_np = np.array(mask)
#         # Crop the image using the mask
#         save_image(image, mask_np, dest, class_id, image_id, i)

def crop_and_save_image(image_path, label_path, dest):
    image = Image.open(image_path)
    image_id = image_path.split('.')[2]
    # Get dimension image
    x, y = image.size
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        # Group coordinates into pairs
        polygon_coords, class_id = get_polygon_coords(line, x, y)

        # Create a mask for the polygon
        mask = create_mask_from_polygon(image.size, polygon_coords, i)

        # Convert mask to numpy array
        mask_np = np.array(mask)
        # Crop the image using the mask
        crop_image = create_crop_image_matrix(image, mask_np)

        # Save the cropped image
        cropped_image_path = f'{dest}/{class_id}_{image_id}_{i}.png'
        if not os.path.exists(dest):
            os.makedirs(dest)
        crop_image.save(cropped_image_path)
        print(f'Cropped image saved to {cropped_image_path}')

def get_polygon_coords(line, x=64, y=64):
    parts = line.strip().split()
    class_id = parts[0]
    coords = list(map(float, parts[1:]))

    # Group coordinates into pairs
    return [(coords[j] * x, coords[j + 1]* y) for j in range(0, len(coords), 2)], class_id

def create_crop_image_matrix(image, mask_np):
    image_np = np.array(image)
    white_background = np.ones_like(image_np) * 255

    # crop image with white background
    masked_image_np = np.where(mask_np[:, :, None] == 1, image_np, white_background)
    crop_boundingbox = get_bounding_box(mask_np)
    masked_image = Image.fromarray(masked_image_np)
    return masked_image.crop(crop_boundingbox)


def extract_non_background_mean_color(img_np):
    # Flatten the image to a 2D array where each row is a pixel's BGR values
    img_flat = img_np.reshape(-1, 3)

    # Create a mask to filter out the background pixels (assuming background is white with all zero values)
    non_background_mask = np.any(img_flat != [255, 255, 255], axis=1)

    # Filter the non-background pixels
    non_background_pixels = img_flat[non_background_mask]

    # Calculate the mean color of the non-background pixels
    if len(non_background_pixels) == 0:
        # If there are no non-background pixels, return a default value (e.g., black)
        return np.array([255, 255, 255])
    else:
        return non_background_pixels.mean(axis=0)



