import os
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

def sign_segmentation(instances, metadata, im, prob_threshold=0.5, segm_size=64):
    '''
    1) checks for the presence of "street_sign" classes according to a given probability threshold;
    2) performs resize according to the given size: 256*256.
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
            im_seg = im[int(y_min - y_delta):int(y_max + y_delta), int(x_min - x_delta):int(x_max + x_delta)]
            im_seg = cv2.resize(im_seg, (segm_size, segm_size))
            segmented_signs.append(im_seg)

    return segmented_signs

def main():
    # Create config
    cfg = get_cfg()
    # Pre-trained model that can detect road signs
    # https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl
    cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

    # Probability threshold for the sought classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45

    # Information about object classes and their labels
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Create predictor
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    test_images = os.listdir('Inference_model/test_images')
    for file in test_images:
        # Load image
        image_url = os.path.join('Inference_model/test_images/', file)
        image = cv2.imread(image_url)

        # Get prediction
        outputs = predictor(image)

        v = Visualizer(image, metadata, scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Filter the drawn instances to include only 'street_sign'
        instances_filtered = outputs["instances"].to("cpu")
        instances_filtered = instances_filtered[
            instances_filtered.pred_classes == metadata.thing_classes.index('street_sign')
        ]

        out = v.draw_instance_predictions(instances_filtered)
        out_image = out.get_image()

        # Save segmented image
        cv2.imwrite(f"Segmentation_image/segmented_images/{file}", out_image)

        # Extract the predicted instances (shards) from outputs:
        instances = outputs["instances"].to("cpu")

        # Get segmented signs
        score_threshold = 0.45
        segmented_signs = sign_segmentation(instances, metadata, image, prob_threshold=score_threshold)

        # Save segmented signs
        for i, sign_segment in enumerate(segmented_signs):
            os.makedirs(f'Segmentation_image/segmented_sign/{file}', exist_ok=True)
            cv2.imwrite(f"Segmentation_image/segmented_sign/{file}/sign_{i}.png", sign_segment)

        print(f'{len(segmented_signs)} traffic signs detected in {file}')

if __name__ == '__main__':
    main()
