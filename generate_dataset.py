from file_utils import list_all_image_file, get_label_path, get_image_path, count_elements_in_class
from image_utils import crop_images


def generate_crop_image(src_image, src_label, dest):
    images = list_all_image_file(src_image)
    for image in images:
        label = get_label_path(src_label, image)
        crop_images(get_image_path(src_image, image), label, dest)
    count_elements_in_class(dest)

generate_crop_image('augment_dataset/test/images', 'augment_dataset/test/labels', './ivyqo/augmented/crop/test')
generate_crop_image('augment_dataset/valid/images', 'augment_dataset/valid/labels', './ivyqo/augmented/crop/valid')
generate_crop_image('augment_dataset/train/images', 'augment_dataset/train/labels', './ivyqo/augmented/crop/train')

