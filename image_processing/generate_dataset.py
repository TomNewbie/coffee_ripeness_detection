from file_utils import list_all_image_file, get_label_path, get_image_path, count_elements_in_class
from image_processing.image_utils import crop_and_save_image


def generate_crop_image(src_image, src_label, dest):
    # function to generate dataset to put into CNN model
    images = list_all_image_file(src_image)
    for image in images:
        label = get_label_path(src_label, image)
        crop_and_save_image(get_image_path(src_image, image), label, dest)
    # count_elements_in_class(dest)

generate_crop_image('dataset/ivyqo/train/images', 'dataset/ivyqo/train/labels', 'dataset/trash/crop/train')
# generate_crop_image('augment_dataset/valid/images', 'augment_dataset/valid/labels', './ivyqo/augmented/crop/valid')
# generate_crop_image('augment_dataset/train/images', 'augment_dataset/train/labels', './ivyqo/augmented/crop/train')

