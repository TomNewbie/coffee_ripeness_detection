import os
from collections import defaultdict

# Specify the directory path
directory_path = 'train/images'
COFFEE_CLASS_IVYQO = ["overripe", "ripe", "semi_ripe", "unripe"]
COFFEE_CLASS_SKEW_DATASET = ['dry', 'overripe', 'ripe', 'semi_ripe', 'unripe']
# List all files and directories in the specified path
import shutil

# Print only the files
def list_all_image_file(image_path):
    images = []
    all_files_and_dirs = os.listdir(image_path)

    for item in all_files_and_dirs:
        path = os.path.join(image_path, item)
        if os.path.isfile(path):
            images.append(item)
            print(item)
    return images

def get_label_path(label_path, image_file):
    image_name, _ = os.path.splitext(image_file)
    return f'{label_path}/{image_name}.txt'

def get_image_path(image_path, image_file_name):
    return os.path.join(image_path, image_file_name)


def count_elements_in_class(path, element_arr = COFFEE_CLASS_IVYQO):
    class_counts = defaultdict(int)

    for filename in os.listdir(path):
        # Split the filename by "_"
        parts = filename.split("_")
        
        # The first part of the filename represents the class
        class_name = parts[0]
        
        # Count the number of elements (filename parts) in this class
        class_counts[class_name] += 1
    total_element = 0
    print(f'number of element in path: {path}: ')
    for class_name, count in class_counts.items():
        total_element += count
        print(f'{element_arr[int(class_name)]} has {count} elements.')
    print(f'total element: {total_element} elemeents')
    print("============================\n\n")
    return class_counts

# count_elements_in_class("ivyqo/crop/train")
# count_elements_in_class("ivyqo/crop/valid")
# count_elements_in_class("ivyqo/crop/test")
# count_elements_in_class("crop/train", COFFEE_CLASS_SKEW_DATASET)
# count_elements_in_class("crop/valid", COFFEE_CLASS_SKEW_DATASET)
# count_elements_in_class("crop/test", COFFEE_CLASS_SKEW_DATASET)
# count_elements_in_class("ivyqo/augmented/crop/train")
# count_elements_in_class("ivyqo/augmented/crop/valid")
# count_elements_in_class("dataset/ivyqo/augmented_mix/crop/train")

def copy_file(src, dest_img, dest_label):
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_label, exist_ok=True)

    images = list_all_image_file(src)
    count = 0 
    for image in images:
        new_file_name = f"background_{count}.jpg"
        shutil.copy2(get_image_path(src, image), get_image_path(dest_img, new_file_name))
        with open(get_label_path(dest_label, new_file_name), 'w') as file:
            pass
        count+=1
# copy_file("dataset/background", "dataset/ivyqo_augment_dataset/train/images", "dataset/ivyqo_augment_dataset/train/labels")