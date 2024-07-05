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
    cluster_image(3, feature_list, images, src_image, c_path)


get_feature_image_from_folder('dataset/trash/test/cluster_0', 'dataset/trash/test/cluster_000')

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


# seperate_img_by_class('dataset/trash/crop/test', 'dataset/trash/test/cluster')
