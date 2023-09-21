from phenobench import PhenoBench
from PIL import Image
import cv2
import numpy as np
import os

# Function to apply the pixel transformation and save the segmentation mask
def process_and_save_image(pixels, image_name):

    result_array = np.where((pixels != 1) & (pixels != 2), 0, pixels)
    result_array[pixels == 1] = 64  # crop
    result_array[pixels == 2] = 255  # weed

    # Convert to uint8 and create a grayscale image
    normalized_array = result_array.astype(np.uint8)
    grayscale_image = cv2.cvtColor(normalized_array, cv2.COLOR_GRAY2BGR)

    # Create the output file path
    output_path = os.path.join("/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/val/segmentation_mask", image_name)

    # Save the segmentation mask
    cv2.imwrite(output_path, grayscale_image)

# Initialize the PhenoBench dataset
val_data = PhenoBench("/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench",
                        target_types=["semantics", "plant_instances", "leaf_instances", "plant_bboxes", "leaf_bboxes"])

# Specify the input and output folders
input_folder = "/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/val/images"
output_folder = "/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/val/segmentation_mask"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


print("Segmentation masks have been saved.")
image_count = len(val_data)
for i in range(image_count):
    pixels = dict(val_data[i].items())["semantics"]
    # print(val_data.filenames[i])
    process_and_save_image(pixels, val_data.filenames[i])
    print("Processed {}/1407 images".format(i, image_count))