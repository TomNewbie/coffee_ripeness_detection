import os
from collections import defaultdict

# Specify the directory path
directory_path = 'train/images'
COFFEE_CLASS_IVYQO = ["overripe", "ripe", "semi_ripe", "unripe"]
COFFEE_CLASS_SKEW_DATASET = ['dry', 'overripe', 'ripe', 'semi_ripe', 'unripe']
# List all files and directories in the specified path

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

def get_image_path(image_path, image_file):
    return os.path.join(image_path, image_file)


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
count_elements_in_class("ivyqo/crop/test")
# count_elements_in_class("crop/train", COFFEE_CLASS_SKEW_DATASET)
# count_elements_in_class("crop/valid", COFFEE_CLASS_SKEW_DATASET)
# count_elements_in_class("crop/test", COFFEE_CLASS_SKEW_DATASET)
# count_elements_in_class("ivyqo/augmented/crop/train")
count_elements_in_class("ivyqo/augmented/crop/valid")
count_elements_in_class("ivyqo/augmented_mix/crop/test")