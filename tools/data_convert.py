
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import tifffile

cfg = None

def convert_to_coco_format(imgs, lbs):
    dataset_dicts = []
    for idx, (img, lb) in enumerate(zip(imgs, lbs)):
        record = {}
        
        height, width = img.shape[:2]
        
        record["file_name"] = f"img_{idx}.png"  # replace with actual file name if available
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        objs = []
        for lb_idx in np.unique(lb):
            if lb_idx == 0:  # typically, 0 is the background
                continue
            
            mask = (lb == lb_idx)
            bbox = np.where(mask)
            
            x_min, y_min = np.min(bbox[1]), np.min(bbox[0])
            x_max, y_max = np.max(bbox[1]), np.max(bbox[0])
            
            obj = {
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": mask.astype(int).tolist(),  # convert to RLE or polygon format for better performance
                "category_id": int(lb_idx),  # replace with actual category id if available
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def convert_tiff_to_coco_format(train_tiff_file, val_tiff_file, train_json_file, val_json_file):
    # Read the tiff file
    train_imgs, train_lbs = tifffile.imread(train_tiff_file[0]), tifffile.imread(train_tiff_file[1])
    val_imgs, val_lbs = tifffile.imread(val_tiff_file[0]), tifffile.imread(val_tiff_file[1])

    # Convert the labels to COCO format
    train_coco = convert_to_coco_format(train_imgs, train_lbs)
    val_coco = convert_to_coco_format(val_imgs, val_lbs)

    # Save the COCO format labels
    print('train_coco segment data type:', type(train_coco[0]['annotations'][0]['segmentation'][0][0]))
    print('type of annotated boxes:', type(train_coco[0]['annotations'][0]['bbox'][0]))
    with open(train_json_file, 'w') as f:
        json.dump(train_coco, f)
    
    with open(val_json_file, 'w') as f:
        json.dump(val_coco, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tiff_file", type=str, default="/content/train.tif")
    parser.add_argument("--train_tiff_gt_file", type=str, default="/content/train_gt.tif")
    parser.add_argument("--val_tiff_file", type=str, default="/content/val.tif")
    parser.add_argument("--val_tiff_gt_file", type=str, default="/content/val_gt.tif")
    parser.add_argument("--train_json_file", type=str, default="/content/updated_labels_2.json")
    parser.add_argument("--val_json_file", type=str, default="/content/updated_labels.json")
    args = parser.parse_args()

    convert_tiff_to_coco_format((args.train_tiff_file, args.train_tiff_gt_file), 
                                (args.val_tiff_file, args.val_tiff_gt_file),
                                 args.train_json_file, 
                                 args.val_json_file)
