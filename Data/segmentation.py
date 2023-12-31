import os
import cv2
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from typing import List, Union

def sign_segmentation(instances: torch.Tensor,
                      metadata: MetadataCatalog,
                      input_image: np.ndarray,
                      segment_size: int = 32,
                      prob_threshold: float = 0.5) -> List[np.ndarray]:
    '''
    Segments signs on the image.
    Parameters:
    - instances: Object detection model results as instances.
    - metadata: Metadata containing information about object classes and additional details.
    - input_image: Input image (NumPy array) for segmentation.
    - prob_threshold: Probability threshold for "street_sign" class detection.
    - segment_size: Size to which the sign segments are resized (default is 32x32).
    Returns a list of segmented sign images (NumPy arrays).
    '''
    # Get indexes of segments corresponding to the "human" class and exceeding the probability threshold
    sign_mask = (instances.pred_classes == metadata.thing_classes.index('street_sign'))   # 'stop_sign'
    high_score_mask = instances.scores > prob_threshold
    selected_mask = sign_mask & high_score_mask

    # If there are suitable segments - process them
    segmented_signs = []
    if selected_mask.any():
        selected_instances = instances[selected_mask]
        for i in range(len(selected_instances)):
            sign_segment = selected_instances[i].to("cpu")
            x_min, y_min, x_max, y_max = sign_segment.pred_boxes.tensor.numpy()[0]
            x_delta = (x_max - x_min) * 0.1
            y_delta = (y_max - y_min) * 0.1
            im_seg = input_image[int(y_min - y_delta):int(y_max + y_delta), int(x_min - x_delta):int(x_max + x_delta)]
            im_seg = cv2.resize(im_seg, (segment_size, segment_size))
            segmented_signs.append(im_seg)

    return segmented_signs

# CUSTOM PARAMETERS
# Shape of segmented signs: [32, 48, 64]
SIZE = 32
# Threshold of probability for segmentation traffic signs
segment_threshold = 0.45
# Path test images folder
PATH_test_imgs = 'raw_images'
# Weights for detectron2 model
detectron2_config_file = "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
DEVICE = "cpu"

def main():
    # Create config (for detectron2)
    cfg = get_cfg()
    # Pre-trained model that can detect road signs
    cfg.merge_from_file(model_zoo.get_config_file(detectron2_config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2_config_file)

    # Probability threshold for the sought classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = segment_threshold

    # Information about object classes and their labels
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Create predictor
    cfg.MODEL.DEVICE = DEVICE
    predictor = DefaultPredictor(cfg)

    test_images = os.listdir(PATH_test_imgs)
    for file in test_images:
        # Load image
        image_url = os.path.join(PATH_test_imgs, file)
        image = cv2.imread(image_url)

        # Get prediction
        outputs = predictor(image)

        # Filter the drawn instances to include only 'street_sign'
        instances_filtered = outputs["instances"].to(DEVICE)
        instances_filtered = instances_filtered[
            instances_filtered.pred_classes == metadata.thing_classes.index('street_sign')
        ]

        # Get image with segmented signs
        v = Visualizer(image, metadata, scale=1.2)
        out = v.draw_instance_predictions(instances_filtered)
        out_image = out.get_image()

        # Save segmented image
        cv2.imwrite(f"segmented_images/{file}", out_image)

        # Extract the predicted instances (shards) from outputs:
        instances = outputs["instances"].to(DEVICE)

        # Get segmented signs
        segmented_signs = sign_segmentation(instances, metadata, image, segment_size=SIZE, prob_threshold=segment_threshold)

        # Save segmented signs
        for i, segmented_sign in enumerate(segmented_signs):
            os.makedirs(f'segmented_signs/model_{SIZE}/{file}', exist_ok=True)
            cv2.imwrite(f'segmented_signs/model_{SIZE}/{file}/sign_{i}.png', segmented_sign)

        print(f'{len(segmented_signs)} traffic signs segmented from {file}')

if __name__ == '__main__':
    main()
