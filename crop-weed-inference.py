import torch, detectron2
import sys, os, distutils.core
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from imutils.video import WebcamVideoStream, FileVideoStream
from imutils.video import FPS
from imutils import resize
import time
from detectron2.utils.visualizer import ColorMode


setup_logger()

model_type = "keypoints"
# model_type = "plant_visibility"
# model_type = "leaf_visibility"

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

def cv2_imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_weed_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# for d in ["train", "val"]:
#     DatasetCatalog.register("weed_" + d, lambda d=d: get_weed_dicts("crop-weed/" + d))
#     MetadataCatalog.get("weed_" + d).set(thing_classes=["weed"])                         

for d in ["train"]:
    DatasetCatalog.register("crop_weed_" + d, lambda d=d: get_weed_dicts("PhenoBench-v100/PhenoBench/{}/semantics".format(d)))
    MetadataCatalog.get("crop_weed_" + d).set(thing_classes=["crop", "weed"])  



weed_metadata = MetadataCatalog.get("crop_weed_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("crop_weed_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.WEIGHTS = os.path.join("detectron2/pre-trained/{}.pth".format(model_type))  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

cfg.MODEL.KEYPOINT_ON = True
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = [0.25, 0.125, 0.0625, 0.03125]

predictor = DefaultPredictor(cfg)
print(cfg)


# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("crop_weed_train", output_dir="./crop_weed_model")
# val_loader = build_detection_test_loader(cfg, "crop_weed_train")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`

cap = WebcamVideoStream(0).start()

# while True:
#     frame = cap.read() 
#     # cv2_imshow(frame)
#     outputs = predictor(frame)  
#     print(outputs["instances"].pred_keypoints)
#     v = Visualizer(frame[:, :, ::-1],
#                     metadata=weed_metadata, 
#                     scale=0.5, 
#                     instance_mode=ColorMode.IMAGE_BW)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("prediction", out.get_image()[:, :, ::-1])
#     cv2.waitKey(1)

import cv2
import numpy as np

while True:
    frame = cap.read() 
    # cv2_imshow(frame)
    outputs = predictor(frame)  
    keypoints = outputs["instances"].pred_keypoints
    
    
    # Get the dimensions of the frame
    height, width, _ = frame.shape
    
    # Calculate the coordinates for the center of the frame
    center_x = width // 2
    center_y = height // 2
    
    # Draw a big red circle on the frame
    radius = 5  # You can adjust the radius as needed
    color = (0, 0, 255)  # Red color in BGR
    thickness = -1  # Fill the circle
    
    for keypoint in keypoints:
        print(keypoint)
        cv2.circle(frame, (int(keypoint[0][0]),int( keypoint[0][1])), radius, color, thickness)
    
    v = Visualizer(frame[:, :, ::-1],
                    metadata=weed_metadata, 
                    scale=1, 
                    )
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(1)





