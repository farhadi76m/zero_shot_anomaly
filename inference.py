import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from eval_tools import get_scores, write_anomaly_to_image, write_anomaly_to_image_no_gt
from datasets import SMIYC
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):
    # Set up models once
    cfg, mask2former_predictor = setup_mask2former(args.config_path, args.model_path)
    sam_predictor = setup_sam(args.sam_checkpoint)

    # Create the directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    gt_values = []
    ignore_values = []
    anomaly_scores = []
    
    dataset = SMIYC(args.dataset)
    dataloader = tqdm(DataLoader(dataset=dataset))
    
    # Process each image in the directory
    # for image_name in os.listdir(args.image_dir):
    for image, gt, ignore, file in dataloader :

        # Run inference and generate outputs
        p, score_map = inference(image, mask2former_predictor)
        boxes, instances = extract_instances_and_boxes(score_map, threshold=-.6, min_size=5000)

        # Generate the anomaly map using SAM and bounding boxes
        anomaly_map = generate_anomaly_map_with_boxes(image, boxes, sam_predictor, score_map)

        # Save the anomaly map as a .npy file for future metrics calculations
        save_path = os.path.join(args.save_dir, "{file[0]}_anomaly_map.npy")
        np.save(save_path, anomaly_map)

        # Visualize anomaly map
        if args.show :
            plt.imshow(anomaly_map, cmap='gray')
            plt.title(f'Anomaly Map: {file[0]}')
            plt.show()
            
        # Evaluate 
        gt_values.append(gt)
        ignore_values.append(ignore)
        anomaly_scores.append(anomaly_map)
    mode = "total"
    ap, roc, fpr, aupr = get_scores(np.asarray(gt_values), np.asarray(anomaly_scores), np.asarray(ignore_values),
                                    mode=mode)
    with open(os.path.join(args.output, "results.txt"), "w") as file:
        file.write(f'{100 * ap:.3f}%\n')
        file.write(f'{100 * roc:.2f}%\n')
        file.write(f'{100 * fpr:.2f}%\n')
        file.write(f'{100 * aupr:.2f}%\n')
        file.write(f'The average precision is: {100 * ap:.3f}%\n')
        file.write(f'The auroc is: {100 * roc:.2f}%\n')
        file.write(f'The fpr is: {100 * fpr:.2f}%\n')
        file.write(f'The aupr is: {100 * aupr:.2f}%\n')

    

if __name__ == "__main__":
    # Setup argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Run anomaly detection and generate anomaly maps.')
    parser.add_argument('config_path', type=str, help='Path to the config file.')
    parser.add_argument('model_path', type=str, help='Path to the model checkpoint file.')
    parser.add_argument('sam_checkpoint', type=str, help='Path to the SAM checkpoint file.')
    parser.add_argument('--image_dir', type=str, help='Directory containing images.')
    parser.add_argument('--save_dir', type=str,  help='Directory to save results.')
    parser.add_argument('--dataset', type=str,help='Choose the dataset you wish to evaluate on.')
    parser.add_argument('--show', action='store_true', help='Online Visualizing.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
