import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from IPython.display import display
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
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
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


from data_convert import convert_tiff_to_coco_format

cfg = None


def train(train_json_file, val_json_file, train_image_dir, val_image_dir, config_file):
    # if your dataset is in COCO format, this cell can be replaced by the following three lines:
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_train_1", {'mask_format': "bitmask"}, train_json_file, train_image_dir)
    register_coco_instances("my_dataset_val_1", {'mask_format': "bitmask"}, val_json_file, val_image_dir)

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("my_dataset_train_1",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg, trainer


def test(cfg, trainer):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    val_dataset_dicts = DatasetCatalog.get("my_dataset_val_1")
    val_metadata = MetadataCatalog.get("my_dataset_val_1")
    

    for d in random.sample(val_dataset_dicts, 1):

        im = cv2.imread(d["file_name"])
        outputs = trainer(im)
        
        v_gt = Visualizer(im[:, :, ::-1], metadata=val_metadata, scale=0.5)
        out_gt = v_gt.draw_dataset_dict(d)
        
        v_pred = Visualizer(im[:, :, ::-1], metadata=val_metadata, scale=0.5) 
        out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))

        f, axarr = plt.subplots(2, 1, figsize=(10, 15))
        
        axarr[0].imshow(out_gt.get_image()[:, :, ::-1])
        axarr[0].set_title('Ground Truth')

        axarr[1].imshow(out_pred.get_image()[:, :, ::-1])
        axarr[1].set_title('Prediction')
        
        plt.tight_layout()
        plt.show()


def evaluate(trainer,cfg):
    evaluator = COCOEvaluator("my_dataset_val_1", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val_1")

    #Use the created predicted model in the previous step
    inference_on_dataset(trainer.model, val_loader, evaluator)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tiff_file", type=str, default="/content/train.tif")
    parser.add_argument("--train_tiff_gt_file", type=str, default="/content/train_gt.tif")
    parser.add_argument("--val_tiff_file", type=str, default="/content/val.tif")
    parser.add_argument("--val_tiff_gt_file", type=str, default="/content/val_gt.tif")
    parser.add_argument("--train_json_file", type=str, default="/content/updated_labels_2.json")
    parser.add_argument("--val_json_file", type=str, default="/content/updated_labels.json")
    parser.add_argument("--train_image_dir", type=str, default="/content/train")
    parser.add_argument("--val_image_dir", type=str, default="/content/val")
    parser.add_argument("--path_config_file", type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    args = parser.parse_args()
    
    # convert_tiff_to_coco_format((args.train_tiff_file, args.train_tiff_gt_file), 
    #                             (args.val_tiff_file, args.val_tiff_gt_file),
    #                              args.train_json_file, 
    #                              args.val_json_file)
    train(
        args.train_json_file,
        args.val_json_file,
        args.train_image_dir,
        args.val_image_dir,
        args.path_config_file
    )