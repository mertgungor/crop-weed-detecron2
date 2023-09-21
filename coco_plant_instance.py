from phenobench import PhenoBench
from PIL import Image
import cv2
import numpy as np
import os
from pprint import pprint
import json

visualise = False
# Initialize the PhenoBench dataset
val_data = PhenoBench("/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench",
                        target_types=["semantics"])
coco_data_dict = {}


image_count = len(val_data)

# Specify the input and output folders
instances_folder = "/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/train/semantics"
output_json_path = "/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/train/semantics/via_region_data.json"
iamge_path = "/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/train/images"

instances = os.listdir(instances_folder)
instances = [instance for instance in instances if instance.endswith(".png")]

print(instances[0])

instances_path = [os.path.join(instances_folder, instance) for instance in instances]
print(instances_path[0]) 

invalid_polygons = 0
valid_polygons = 0

for i in range(image_count):
    pixels = dict(val_data[i].items())["semantics"]

    image_name = val_data[i]["image_name"]
    image_path = os.path.join(iamge_path, image_name)

    # print(val_data.filenames[i])
    print("Processed {}/1407 images".format(i, image_count))

    # print(np.unique(pixels))

    image_info = {
        "fileref": "",
        "size": os.path.getsize(image_path),
        "filename": image_path ,
        "base64_img_data": "",
        "file_attributes": {},
        "regions": {}
    }
    annotation_id = 0

    for j in [1, 2]:
        # print("i: ", i)
        target_value = j
        # normalized_array = pixels.astype(np.uint8)
        # Create a binary mask where the target value is set to 255 (white) and other values are set to 0 (black)
        valid_values = [0, 1, 2]

        pixels = np.where(np.isin(pixels, valid_values), pixels, 0)

        binary_image = np.where(pixels == target_value, 255, 0).astype(np.uint8)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if visualise:
            # Draw the contours on a copy of the original image for visualization
            contour_image = np.copy(pixels)
            color = 255 if i == 1 else 64
            cv2.drawContours(contour_image, contours, -1, color, 2)
            cv2.imshow("Original Image", pixels.astype(np.uint8))
            cv2.imshow("Contour Image", contour_image.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Process each contour to create annotations
        for contour in contours:
            # Extract the x and y coordinates separately
            x_coords = contour[:, 0, 0].tolist()
            y_coords = contour[:, 0, 1].tolist()

            if (not len(x_coords) < 5 and not len(y_coords) < 5):

                # Classify as "weed" (64) or "crop" (255)
                if   j == 1:
                    region_attributes = {"name": "crop"}
                elif j == 2:
                    region_attributes = {"name": "weed"}
                
                center_x = (np.max(x_coords) + np.min(x_coords))/2
                center_y = (np.max(y_coords) + np.min(y_coords))/2

                # Create annotation information
                annotation_info = {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": x_coords,
                        "all_points_y": y_coords,
                        "center_x": center_x,
                        "center_y": center_y
                    },
                    "region_attributes": region_attributes
                }

                # Add annotation to COCO data
                image_info["regions"][str(annotation_id)] = annotation_info

                annotation_id += 1
                valid_polygons += 1

            else:
                print("x: ", x_coords)
                print("y: ", y_coords)
                invalid_polygons += 1
                print()

        # Store the COCO data for this image in the dictionary with the filename as the key
    coco_data_dict[image_name] = image_info

print("Invalid polygons: ", invalid_polygons)
print("Valid polygons: ", valid_polygons)

with open(output_json_path, 'w') as json_file:
    json.dump(coco_data_dict, json_file)


