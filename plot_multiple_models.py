import random
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from architectures import (
    MLP_Optuna,
    Conv1DNet_Optuna,
    CNNLSTMNet_Optuna,
    CNNTransformerNet_Optuna,
)
from testing_model import read_odom_csv, read_scan_csv, run_inference, load_model
from utils.utils import calc_accuracy_percentage_xy
#######################################
# Logging Setup
#######################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

#######################################
# Accuracy and Visualization
#######################################
def evaluate_models(models, model_names, scan_data, odom_data, device='cpu', sample_fraction=0.015):
    """
    Evaluates multiple models and plots their predictions vs ground truth.
    Args:
        models: list of model objects
        model_names: list of model names
        scan_data: list of LiDAR scan dicts
        odom_data: dictionary of ground truth odometry
        device: 'cpu' or 'cuda'
        sample_fraction: fraction of the dataset to use for evaluation (e.g., 0.1 for 10%)
    """
    # Select a sequential subset of the data
    num_samples = max(1, int(len(scan_data) * sample_fraction))
    sampled_scan_data = scan_data[:num_samples]
    logger.info(f"Evaluating on {num_samples}/{len(scan_data)} samples ({sample_fraction * 100:.1f}%)")

    results = {}
    for model, name in zip(models, model_names):
        logger.info(f"Evaluating model: {name}")

        gt_list = []
        pred_list = []

        for scan in sampled_scan_data:
            key = scan["key"]
            ranges = scan["ranges"]

            if key in odom_data:
                x_gt = odom_data[key]["x"]
                y_gt = odom_data[key]["y"]
                yaw_gt = odom_data[key]["yaw"]

                x_pred, y_pred, yaw_pred = run_inference(model, ranges, device)

                gt_list.append([x_gt, y_gt, yaw_gt])
                pred_list.append([x_pred, y_pred, yaw_pred])

        # Convert lists to arrays for accuracy calculation
        gt_array = np.array(gt_list)
        pred_array = np.array(pred_list)

        # Calculate accuracy
        accuracy, x_thresh, y_thresh = calc_accuracy_percentage_xy(gt_array, pred_array)
        results[name] = {"accuracy": accuracy, "gt_array": gt_array, "pred_array": pred_array}

        logger.info(f"Model: {name} | Accuracy: {accuracy:.2f}% "
                    f"(x_thresh={x_thresh}, y_thresh={y_thresh})")

    # Plot predictions vs ground truth
    plot_predictions(results)


def plot_predictions(results):
    """
    Plots predictions vs ground truth for each model with lines connecting points.
    Args:
        results: dictionary containing model accuracy and predictions.
    """
    plt.figure(figsize=(15, 10))

    for idx, (model_name, data) in enumerate(results.items()):
        gt_array = data["gt_array"]
        pred_array = data["pred_array"]
        accuracy = data["accuracy"]

        plt.subplot(2, 2, idx + 1)

        # Plot ground truth points
        plt.scatter(
            gt_array[:, 0], gt_array[:, 1],
            color='blue', label='Ground Truth', alpha=0.6, s=10
        )

        # Plot predicted points
        plt.scatter(
            pred_array[:, 0], pred_array[:, 1],
            color='red', label='Predicted', alpha=0.6, s=10
        )

        # Connect ground truth and predictions with lines
        for i in range(len(gt_array)):
            plt.plot(
                [gt_array[i, 0], pred_array[i, 0]],
                [gt_array[i, 1], pred_array[i, 1]],
                color='gray', linestyle='--', alpha=0.5, linewidth=0.8
            )

        plt.title(f"{model_name} (Accuracy: {accuracy:.2f}%)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()


#######################################
# Main Execution
#######################################
if __name__ == "__main__":
    # File paths (replace with actual paths)
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"

    # Read data
    odom_data = read_odom_csv(odom_csv_path)
    scan_data = read_scan_csv(scan_csv_path)

    # Define models and their saved paths
    model_choices = [
        ("MLP_Optuna", "models/MLP_Optuna_lr0.000137_bs16_20250127_142609.pth"),
        ("Conv1DNet_Optuna", "models/Conv1DNet_Optuna_lr0.002_bs16_20250127_152439.pth"),
        ("CNNLSTMNet_Optuna", "models/CNNLSTMNet_Optuna_lr0.0006829381720401536_bs16_20250127_144344.pth"),
        ("CNNTransformerNet_Optuna", "models/CNNTransformerNet_Optuna_lr6.89e-05_bs16_20250127_131437.pth"),
    ]

    models = []
    model_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, model_path in model_choices:
        logger.info(f"Loading model: {model_name}")
        model = load_model(model_path, model_name, device=device)
        models.append(model)
        model_names.append(model_name)

    # Evaluate and plot
    evaluate_models(models, model_names, scan_data, odom_data, device=device, sample_fraction = 0.1)
