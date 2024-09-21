import cv2
import numpy as np
from easydict import EasyDict as edict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from segment_anything import SamPredictor, sam_model_registry

# Function to set up the Mask2Former model configuration and load the weights
def setup_mask2former(cfg_path, model_path):
    """
    Sets up Mask2Former model configuration.
    """
    args = edict({'config_file': cfg_path, 'eval-only': True, 'opts': ["MODEL.WEIGHTS", model_path]})

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg, DefaultPredictor(cfg)

# Function to load SAM model (once)
def setup_sam(sam_checkpoint):
    """
    Sets up the SAM model.
    """
    sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint)
    sam.to("cuda")
    return SamPredictor(sam)

# Function to perform inference on an image
def inference(image, predictor):
    """
    Runs inference on a given image using Mask2Former predictor.
    """
    outputs = predictor(image)
    score_map = -outputs['sem_seg'].sum(0).tanh().cpu().numpy()
    return outputs, score_map

# Function to extract instances from the score map
def extract_instances(score_map, threshold=0.5, min_size=10):
    _, binary_map = cv2.threshold(score_map, threshold, 1, cv2.THRESH_BINARY)
    binary_map = binary_map.astype(np.uint8)
    num_labels, labeled_map, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    instances = []
    filtered_centroids = []
    filtered_labeled_map = np.zeros_like(labeled_map, dtype=np.uint8)

    for label in range(1, num_labels):
        size = stats[label, cv2.CC_STAT_AREA]
        if size >= min_size:
            instance_mask = (labeled_map == label).astype(np.uint8)
            instances.append(instance_mask)
            filtered_labeled_map[labeled_map == label] = label
            filtered_centroids.append(centroids[label])

    return instances, filtered_labeled_map, filtered_centroids

# Function to extract the road mask from the logits
def extract_road_mask(p, kernel_size=(300, 300)):
    logit = p['sem_seg'].argmax(0).cpu()
    road_mask = np.where(logit == 0, 1, 0).astype(np.float64)
    kernel = np.ones(kernel_size, np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    return road_mask

# Function to generate anomaly map using SAM and road information
def generate_anomaly_map(image, centroids, road, sam_predictor, score_map):
    sam_predictor.set_image(image)
    anomaly_map = np.zeros_like(score_map)

    for i, c in enumerate(centroids):
        if road[int(c[1]), int(c[0])] > 0:
            input_point = np.array([centroids[i]])
            input_label = np.arange(1, len([centroids[i]]) + 1)

            masks, scores, logits = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            a = masks[scores.argmax()]
            anomaly_map = np.maximum(a, anomaly_map)

    return anomaly_map

# Function to extract instances and bounding boxes from the score map
def extract_instances_and_boxes(score_map, threshold=0.8, min_size=10):
    """
    Extract instances and their bounding boxes from the anomaly score map.

    Parameters:
    - score_map: 2D numpy array, anomaly score map (e.g., probability map)
    - threshold: float, threshold to convert score map into a binary map
    - min_size: int, minimum size (in pixels) to keep an instance

    Returns:
    - boxes: List of tuples, each tuple containing (x, y, w, h) for the bounding box
    - instances: List of 2D numpy arrays, each representing a binary mask of an instance
    """
    # Step 1: Threshold the score map
    _, binary_map = cv2.threshold(score_map, threshold, 1, cv2.THRESH_BINARY)
    binary_map = binary_map.astype(np.uint8)

    # Step 2: Connected component labeling using OpenCV
    num_labels, labeled_map, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    # Step 3: Extract instances and bounding boxes
    instances = []
    boxes = []
    for label in range(1, num_labels):  # Skip label 0 (background)
        size = stats[label, cv2.CC_STAT_AREA]
        if size >= min_size:
            instance_mask = (labeled_map == label).astype(np.uint8)
            instances.append(instance_mask)

            # Extract bounding box (x, y, w, h) for the instance
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                         stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, w, h))

    return boxes, instances

# Function to generate anomaly map using SAM and bounding boxes
def generate_anomaly_map_with_boxes(image, boxes, sam_predictor, score_map):
    """
    Uses SAM to generate an anomaly map based on bounding boxes.

    Parameters:
    - image: Input image
    - boxes: List of bounding boxes (x, y, w, h) for detected instances
    - sam_predictor: SAM model predictor
    - score_map: The score map from segmentation model output

    Returns:
    - anomaly_map: 2D numpy array representing the combined anomaly map
    """
    # Set the input image for the SAM model
    sam_predictor.set_image(image)

    # Initialize an empty anomaly map
    anomaly_map = np.zeros_like(score_map)

    # Loop through bounding boxes and apply SAM for each bounding box
    for box in boxes:
        x, y, w, h = box
        input_box = np.array([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2] format for SAM

        # Predict using SAM
        masks, scores, logits = sam_predictor.predict(
            box=input_box,
            multimask_output=True
        )

        # Select the mask with the highest score
        best_mask = masks[scores.argmax()]

        # Update the anomaly map with the mask
        print(best_mask.shape)
        anomaly_map = np.maximum(best_mask, anomaly_map)

    return anomaly_map


def heatmap(image, probs):
    heatmap_img = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    return super_imposed_img


def heatmap(image, probs):
    heatmap_img = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    return super_imposed_img