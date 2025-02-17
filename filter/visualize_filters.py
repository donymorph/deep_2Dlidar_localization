import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import model architectures and dataset
from architectures.architectures import (
    SimpleMLP, MLP_Optuna,
    Conv1DNet, Conv1DNet_Optuna,
    Conv1DLSTMNet, CNNLSTMNet_Optuna,
    ConvTransformerNet, CNNTransformerNet_Optuna
)

from dataset import LidarOdomDataset
from torch.utils.data import DataLoader
from utils import utils
from filter.filters import apply_kalman_filter_3d, apply_extended_kalman_filter_3d, apply_unscented_kalman_filter_3d, apply_particle_filter_3d

def get_model(model_choice: str, input_size: int):
    if model_choice == 'MLP_Optuna':
        model = MLP_Optuna(input_size=input_size, output_size=3)
    elif model_choice == 'Conv1DNet_Optuna':
        model = Conv1DNet_Optuna(input_size=input_size, output_size=3)
    elif model_choice == 'CNNLSTMNet_Optuna':
        model = CNNLSTMNet_Optuna(input_size=input_size, output_size=3)
    elif model_choice == 'CNNTransformerNet_Optuna':
        model = CNNTransformerNet_Optuna(output_size=3)
    elif model_choice == 'ConvTransformerNet':
        model = ConvTransformerNet(input_size=input_size, output_size=3)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")
    return model

def generate_predictions(model: torch.nn.Module, batch_size: int, dataset, device="cuda") -> np.ndarray:
    """
    Generates predictions for the entire dataset (or subset) in memory.
    Returns a NumPy array with shape (N, 3): [pos_x, pos_y, orientation_z].
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for lidar_batch, _ in data_loader:
            preds = model(lidar_batch.to(device))
            all_preds.append(preds.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return all_preds


def visualize_results(odom_gt: np.ndarray, preds: np.ndarray, filtered: np.ndarray,
                      model_acc: float, kalman_acc: float, max_samples: int, thresh_x: float, thresh_y: float):
    """
    Visualizes:
      1. Trajectories (position x vs. y) for ground truth, model predictions, and Kalman-filtered predictions.
      2. A bar chart comparing the accuracy percentages.
    """
    plt.figure(figsize=(14, 6))
    
    # Trajectory Comparison Plot
    plt.subplot(1, 2, 1)
    plt.plot(odom_gt[:, 0], odom_gt[:, 1], 'k-', label='Ground Truth')
    plt.plot(preds[:, 0], preds[:, 1], 'r--', label='Model Prediction')
    plt.plot(filtered[:, 0], filtered[:, 1], 'b-.', label='Kalman Filtered')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Bar Chart
    plt.subplot(1, 2, 2)
    labels = ['Model Prediction', 'Kalman Filtered']
    accuracies = [model_acc, kalman_acc]
    plt.bar(labels, accuracies, color=['red', 'blue'])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 2, f"{acc:.2f}%", ha='center', fontsize=12)
    
    text = f"Max Samples: {max_samples}\nThresh X: {thresh_x}m\nThresh Y: {thresh_y}m"
    plt.gca().text(0.95, 0.95, text, horizontalalignment='right', verticalalignment='top',
                   transform=plt.gca().transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/CNNTransformerNet_Optuna_lr6.89e-05_bs16_20250127_131437.pth",
                        help="Path to the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for prediction")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum number of samples to use from the dataset")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------- LOAD THE MODEL ------------------------------
    model_params = os.path.basename(opt.model_path)
    # Expected model_params: "CNNTransformerNet_Optuna_lr6.89e-05_bs16_20250127_131437.pth"
    parts = model_params.split('_')
    model_choice = parts[0]
    optim_str = parts[1]
    lr = parts[2]
    bs = parts[3]
    batch_size = opt.batch_size

    input_size = 360  # Number of LiDAR beams
    model = get_model(model_choice=f"{model_choice}_{optim_str}", input_size=input_size)
    model.to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device), strict=False)
    odom_data_path = "dataset/odom_data.csv"
    scan_data_path = "dataset/scan_data.csv"
    # ---------------- LOAD DATASET ------------------------------
    # Using the same CSV for odometry and scan data (as specified)
    dataset = LidarOdomDataset(odom_csv_path=odom_data_path, scan_csv_path=scan_data_path)

    # Limit dataset to first max_samples samples if necessary
    if len(dataset) > opt.max_samples:
        dataset.data = dataset.data.iloc[:opt.max_samples]
    
    # Ground truth odometry: columns [pos_x, pos_y, orientation_z]
    odom_gt = dataset.data[['pos_x', 'pos_y', 'orientation_z']].to_numpy()
    
    # ---------------- GENERATE MODEL PREDICTIONS --------------------
    preds = generate_predictions(model=model, batch_size=batch_size, dataset=dataset, device=device)
    
    # ---------------- APPLY KALMAN FILTER --------------------------

    filtered = apply_kalman_filter_3d(preds)
    
    # ---------------- CALCULATE ACCURACY ---------------------------
    model_acc, _, _ = utils.calc_accuracy_percentage_xy(gt_array=odom_gt, pred_array=preds)
    kalman_acc, thresh_x, thresh_y = utils.calc_accuracy_percentage_xy(gt_array=odom_gt, pred_array=filtered)
    
    print("Accuracy Comparison:")
    print(f"Model Predictions: {model_acc:.2f}%")
    print(f"Kalman Filtered:   {kalman_acc:.2f}%")
    
    # ---------------- VISUALIZE RESULTS ---------------------------
    visualize_results(odom_gt, preds, filtered, model_acc, kalman_acc, opt.max_samples, thresh_x, thresh_y)

if __name__ == "__main__":
    main()
