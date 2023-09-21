import torch, detectron2
import os
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

setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

category_to_id = {
    "crop": 0,
    "weed": 1,
}

def cv2_imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_weed_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # print(list(imgs_anns.values())[0])

    dataset_dicts = []

    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        # print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():

            region_attributes = anno["region_attributes"]
            # print(region_attributes)
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            center_x = anno["center_x"]
            center_y = anno["center_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            if(len(px)==4):
                print("=========================================================")
                print(px)
                print("=========================================================")

            if region_attributes:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": category_to_id[region_attributes["name"]],
                    "keypoints": [center_x, center_y, 1],
                    
                }

                objs.append(obj)
        record["annotations"] = objs
        
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train"]:

    DatasetCatalog.register("crop_weed_" + d, lambda d=d: get_weed_dicts("PhenoBench-v100/PhenoBench/" + d + "/semantics"))
    MetadataCatalog.get("crop_weed_" + d).set(thing_classes=["crop", "weed"])
    MetadataCatalog.get("crop_weed_" + d).set(keypoint_names=["stem"])
    MetadataCatalog.get("crop_weed_" + d).set(keypoint_flip_map=[])


weed_metadata = MetadataCatalog.get("crop_weed_train")

print(weed_metadata.thing_classes)
dataset_dicts = get_weed_dicts("PhenoBench-v100/PhenoBench/train/semantics")

random_numbers = [random.randint(0, 1406) for _ in range(3)]
for i in random_numbers:
    d = dataset_dicts[i]
    img = cv2.imread(d["file_name"])
    segmentation_path = os.path.join("/home/mert/phenobench-baselines/PhenoBench-v100/PhenoBench/train/segmentation_mask", d["file_name"].split("/")[-1])
    visualizer = Visualizer(img[:, :, ::-1], metadata=weed_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    segmentation_img = cv2.imread(segmentation_path)
    cv2.imshow('seg', segmentation_img)
    cv2.imshow('img', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cfg = get_cfg()
cfg.OUTPUT_DIR = "./detectron2/crop_weed_model"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("crop_weed_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 1000    
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  

# keypoints
cfg.MODEL.KEYPOINT_ON = True
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = [0.25, 0.125, 0.0625, 0.03125]
# cfg.INPUT.MASK_FORMAT = 'bitmask'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()