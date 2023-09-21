import os
import cv2
import numpy as np
import json

d = ["train", "val"]
# Specify the folder containing the mask images and where you want to save the output JSON

# model = "plant"
model = "leaf"

for folder in d:
    mask_folder = 'PhenoBench-v100/PhenoBench/{}/{}_visibility'.format(folder, model)
    output_json_path = 'PhenoBench-v100/PhenoBench/{}/{}_visibility/via_region_data.json'.format(folder, model)

    # Initialize a dictionary to store the COCO data
    coco_data_dict = {}

    # Iterate through the mask files in the folder
    for filename in os.listdir(mask_folder):
        if filename.endswith(".png"):
            # Load the PNG image mask
            mask_path = os.path.join(mask_folder, filename)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_COLOR)

            # Convert the mask to binary
            binary_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize COCO JSON structure for this image
            image_info = {
                "fileref": "",
                "size": os.path.getsize(mask_path),
                "filename": "/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/{}/images/".format(folder) + filename,
                "base64_img_data": "",
                "file_attributes": {},
                "regions": {}
            }

            # Process each contour to create annotations
            annotation_id = 0
            for contour in contours:
                # Extract the x and y coordinates separately
                x_coords = contour[:, 0, 0].tolist()
                y_coords = contour[:, 0, 1].tolist()

                if(not len(x_coords)<5 and not len(y_coords)<5):

                    # Create annotation information
                    annotation_info = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": x_coords,
                            "all_points_y": y_coords
                        },
                        "region_attributes": {"weed": "weed"}
                    }

                    # Add annotation to COCO data
                    image_info["regions"][str(annotation_id)] = annotation_info

                    annotation_id += 1
                else:
                    print("x: ", x_coords)
                    print("y: ", y_coords)
                    print()


            # Store the COCO data for this image in the dictionary with the filename as the key
            coco_data_dict[filename] = image_info

    # Save the JSON dictionary to a file
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data_dict, json_file)
