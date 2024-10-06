from PIL import Image, ImageDraw, ImageFont
from file_utils import get_file_name_by_path 

SPACE_VER = 10
SPACE_HOR = 30

def save_predict_and_groundtruth(predict_path, gt_path, dest):
    # Load the images   
    predicted_image = Image.open(predict_path)
    ground_truth_image = Image.open(gt_path)

    # Get dimensions of the images
    pred_width, pred_height = predicted_image.size
    gt_width, gt_height = ground_truth_image.size


    # Create a new image with enough SPACE_VER to hold both images side by side
    combined_width = pred_width + gt_width + SPACE_VER
    combined_height = max(pred_height, gt_height) + SPACE_HOR
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the images onto the combined canvas
    combined_image.paste(predicted_image, (0, SPACE_HOR))
    combined_image.paste(ground_truth_image, (pred_width + SPACE_VER, SPACE_HOR))

    # Draw a vertical line to separate the two images
    draw = ImageDraw.Draw(combined_image)
    line_x = pred_width + SPACE_VER / 2  # x-coordinate of the vertical line
    draw.line([(line_x, 0), (line_x, combined_height)], fill="red", width=5)

    # set text
    font = ImageFont.load_default(size=25)
    predict_text_location = (pred_width / 2, 0)
    draw.text(predict_text_location, "Predict", fill="white", font=font)

    true_text_location = (pred_width + SPACE_VER + gt_width/2, 0)
    class_test = f"category_id"
    draw.text(true_text_location, "GroundTrue", fill="white", font=font)

    # Save or display the combined image
    file_name = get_file_name_by_path(gt_path)
    file_path = f"{dest}/combine_{file_name}"
    combined_image.save(file_path)
    combined_image.show()
