
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

from pycocotools import mask as mask_util

import tifffile
import base64

cfg = None

def convert_to_coco_format(imgs, lbs):
    coco_json = {}
    coco_json['images'] = []
    coco_json['annotations'] = []
    coco_json['categories'] = []

    cats = []
    ann_id = 0
    for idx, (img, lb) in enumerate(zip(imgs, lbs)):
        record = {}
        record["file_name"] = f"img_{idx}.png"
        record["id"] = idx
        record["height"] = img.shape[0]
        record["width"] = img.shape[1]
        coco_json['images'].append(record)

        annots = []
        
        for lb_idx in np.unique(lb):
            cats.append(lb_idx)
            if lb_idx == 0:  # typically, 0 is the background
                continue
            
            mask = (lb == lb_idx).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Create mask for current contour
                contour_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=cv2.FILLED)

                # Encode mask using RLE
                encoded_mask = mask_util.encode(np.asfortranarray(contour_mask))
                rle_str = base64.b64encode(encoded_mask['counts']).decode("utf-8")

                # Calculate bounding box from contour
                x, y, w, h = cv2.boundingRect(contour)

                # Create annotation dictionary
                ann_id += 1
                annot = {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "bbox_mode": "BoxMode.XYWH_ABS",
                    "segmentation": rle_str,
                    "category_id": int(lb_idx), 
                    "image_id": idx,
                    "id": ann_id
                }
                annots.append(annot)
        coco_json['annotations'].extend(annots)
    print('np.unique(cats):', np.unique(cats))
    for cat in np.unique(cats):
        if cat == 0:
            continue
        coco_json['categories'].append({'id': int(cat), 'name': f"cat_{cat}", 'supercategory': 'cat'})
    return coco_json


def convert_tiff_to_coco_format(train_tiff_file, val_tiff_file, train_json_file, val_json_file):
    # Read the tiff file
    train_imgs, train_lbs = tifffile.imread(train_tiff_file[0]), tifffile.imread(train_tiff_file[1])
    val_imgs, val_lbs = tifffile.imread(val_tiff_file[0]), tifffile.imread(val_tiff_file[1])

    # Convert the labels to COCO format
    train_coco = convert_to_coco_format(train_imgs, train_lbs)
    val_coco = convert_to_coco_format(val_imgs, val_lbs)

    # Save the COCO format labels
    with open(train_json_file, 'w') as f:
        json.dump(train_coco, f)
    
    with open(val_json_file, 'w') as f:
        json.dump(val_coco, f)


def write_images_from_tiff(train_tiff_file, val_tiff_file, train_images_dir, val_images_dir):
    # Read the tiff file
    train_imgs = tifffile.imread(train_tiff_file)
    val_imgs = tifffile.imread(val_tiff_file)

    # Write the images
    for idx, img in enumerate(train_imgs):
        cv2.imwrite(os.path.join(train_images_dir, f"img_{idx}.png"), img)
    
    for idx, img in enumerate(val_imgs):
        cv2.imwrite(os.path.join(val_images_dir, f"img_{idx}.png"), img)


def convert_jb_to_coco_json(image_folder, masks_folder, coco_json_file):
    
    import glob
    coco_json = {}
    coco_json['images'] = []
    coco_json['annotations'] = []
    coco_json['categories'] = []

    ann_id = 0
    imid = 0
    for gl in glob.glob(os.path.join(image_folder, "*.jpg")):
        # get the name somename.input.jpg
        name = os.path.basename(gl)
        # remove the .input.jpg
        name = name[:-10]
        
        # get the mask file in format somename.output.jpg
        mask_file = os.path.join(masks_folder, name + ".output.jpg")

        # read the mask file
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print('mask is None. file:', mask_file)
            continue

        # convert to binary
        mask = (mask > 0).astype(np.uint8)

        # coco annotation format
        imid += 1
        record = {}
        record["file_name"] = name + ".input.jpg"
        record["id"] = imid
        record["height"] = mask.shape[0]
        record["width"] = mask.shape[1]
        coco_json['images'].append(record)
        
        # get the mask
        objs = []
        for lb_idx in np.unique(mask):
            if lb_idx == 0:
                continue
            mask_ = (mask == lb_idx).astype(np.uint8)
            contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Create mask for current contour
                contour_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=cv2.FILLED)

                # Encode mask using RLE
                encoded_mask = mask_util.encode(np.asfortranarray(contour_mask))
                rle_str = base64.b64encode(encoded_mask['counts']).decode("utf-8")

                # Calculate bounding box from contour
                x, y, w, h = cv2.boundingRect(contour)

                # Create annotation dictionary
                ann_id += 1
                obj = {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "bbox_mode": "BoxMode.XYWH_ABS",
                    "segmentation": rle_str,
                    "category_id": int(lb_idx),
                    "image_id": imid,
                    "id": ann_id
                }
                objs.append(obj)
        coco_json['annotations'].extend(objs)

    # add the categories
    coco_json['categories'].append({'id': 0, 'name': 'bg', 'supercategory': 'background'})
    coco_json['categories'].append({'id': 1, 'name': 'fg', 'supercategory': 'foreground'})

    # save the json file
    with open(coco_json_file, 'w') as f:
        json.dump(coco_json, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # cell dataset
    parser.add_argument("--train_tiff_file", type=str, default="/content/train.tif")
    parser.add_argument("--train_tiff_gt_file", type=str, default="/content/train_gt.tif")
    parser.add_argument("--val_tiff_file", type=str, default="/content/val.tif")
    parser.add_argument("--val_tiff_gt_file", type=str, default="/content/val_gt.tif")
    parser.add_argument("--train_json_file", type=str, default="/content/updated_labels_2.json")
    parser.add_argument("--val_json_file", type=str, default="/content/updated_labels.json")
    parser.add_argument("--train_images_dir", type=str, default="/content/train")
    parser.add_argument("--val_images_dir", type=str, default="/content/val")

    # jb dataset
    parser.add_argument("--jb_image_folder", type=str, default="/content/jb/images")
    parser.add_argument("--jb_masks_folder", type=str, default="/content/jb/masks")
    args = parser.parse_args()

    # convert_tiff_to_coco_format((args.train_tiff_file, args.train_tiff_gt_file), 
    #                             (args.val_tiff_file, args.val_tiff_gt_file),
    #                              args.train_json_file, 
    #                              args.val_json_file)
    # write_images_from_tiff(args.train_tiff_file, args.val_tiff_file, args.train_images_dir, args.val_images_dir)

    convert_jb_to_coco_json(args.jb_image_folder, args.jb_masks_folder, args.train_json_file)
