from image_utils import extract_non_background_mean_color, create_mask_from_polygon, get_polygon_coords, \
    create_crop_image_matrix
from file_utils import list_all_image_file, get_image_path
import os
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import numpy as np
import shutil


def cluster_image(num_clusters, features, images, image_path, c_path):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    clusters = {i: [] for i in range(num_clusters)}

    for img, label in zip(images, labels):
        clusters[label].append(get_image_path(image_path, img))
    for cluster, paths in clusters.items():
        cluster_path = f'{c_path}_{cluster}'
        os.makedirs(cluster_path, exist_ok=True)
        for path in paths:
            img_name = os.path.basename(path)
            new_path = os.path.join(cluster_path, img_name)
            shutil.copy2(path, new_path)


def get_feature_image(image_path, label_path, dest):
    image = Image.open(image_path)
    image_id = image_path.split('.')[2]

    # Get dimension image
    x, y = image.size
    with open(label_path, 'r') as file:
        lines = file.readlines()
    feature_list = []
    for i, line in enumerate(lines):
        polygon_coords, class_id = get_polygon_coords(line, x, y)
        # Create a mask for the polygon
        mask = create_mask_from_polygon(image.size, polygon_coords, i)

        # Convert mask to numpy array
        mask_np = np.array(mask)
        feature_list.append(extract_non_background_mean_color(mask_np))
    return feature_list


def get_feature_image_from_folder(src_image, c_path):
    images = list_all_image_file(src_image)
    feature_list = []
    for image in images:
        image_np = np.array(Image.open(get_image_path(src_image, image)))
        feature_list.append(extract_non_background_mean_color(image_np))
    cluster_image(5, feature_list, images, src_image, c_path)


get_feature_image_from_folder('dataset/trash/testdataset/ripe', 'dataset/trash/testdataset/split/ripe')

def seperate_img_by_class(src_image, dest):
    for class_id in range(4):
        cluster_path = f'{dest}_{class_id}'
        os.makedirs(cluster_path, exist_ok=True)

    switcher = {
        0: f'{dest}_0',
        1: f'{dest}_1',
        2: f'{dest}_2',
        3: f'{dest}_3'
    }
    images = list_all_image_file(src_image)
    for image in images:
        image_path= get_image_path(src_image, image)
        img_name = os.path.basename(image_path)
        class_id = img_name.split('_')[0]
        new_path = os.path.join(switcher.get(int(class_id)), img_name)
        shutil.copy2(image_path, new_path)

import cv2
import numpy as np
# seperate_img_by_class('dataset/ivyqo/augmented_mix/crop/test', 'dataset/trash/test/another_test')
#-10, 160, 100 -> 10, 255,175
def classify_color(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for red color in HSV
    lower_red1 = np.array([0, 160, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 160, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([25, 255, 49])

    # Create masks for red and black colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Count the number of pixels for each color
    red_pixels = cv2.countNonZero(mask_red)
    black_pixels = cv2.countNonZero(mask_black)
    return red_pixels, black_pixels
    
    #     color = "Red"
    # else:
    #     color = "Black"

    # # Print the result
    # print(f"The object is classified as: {color}")

    # # Optionally, display the masks for visual verification
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Red Mask', mask_red)
    # cv2.imshow('Black Mask', mask_black)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def seperate_img_by_BLACK(src_image, dest):
    os.makedirs(f"{dest}/black", exist_ok=True)
    os.makedirs(f"{dest}/red", exist_ok=True)

    list_image_file_name = list_all_image_file(src_image)
    for image_name in list_image_file_name:
        image_path= get_image_path(src_image, image_name)
        red_pixels, black_pixels = classify_color(image_path)
        # Determine the color with the most pixels
        if red_pixels <= black_pixels:
            shutil.copy2(image_path, os.path.join(f"{dest}/black", image_name))
        else:
            shutil.copy2(image_path, os.path.join(f"{dest}/red", image_name))

# seperate_img_by_BLACK('dataset/trash/another_test_0', 'dataset/trash/test/red_split')