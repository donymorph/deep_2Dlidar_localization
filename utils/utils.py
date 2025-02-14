import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# ---------------------------
# Setup Python logging
# ---------------------------

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Only add a handler if it doesn't already exist
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
# ---------------------------
# Check GPU
# ---------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# TensorBoard Setup
# ---------------------------
def setup_tensorboard(log_dir, model_choice, lr, batch_size):
    if log_dir is None:
        log_dir = f"tensorboard_logs/{model_choice}_lr{lr}_bs{batch_size}"
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def calc_accuracy_percentage_xy(
    gt_array: np.ndarray,
    pred_array: np.ndarray,
    x_thresh: float = 0.1,
    y_thresh: float = 0.1
) -> float:
    """
    Compares ground truth vs. predicted 2D positions in arrays of shape (N,2).
    Returns the percentage of samples within x_thresh, y_thresh.
    """
    N = gt_array.shape[0]
    if N == 0:
        return 0.0

    correct = 0
    for i in range(N):
        x_gt, y_gt, yaw_gt = gt_array[i]
        x_pd, y_pd, yaw_gt = pred_array[i]
        err_x = abs(x_pd - x_gt)
        err_y = abs(y_pd - y_gt)
        if err_x <= x_thresh and err_y <= y_thresh:
            correct += 1
    accuracy = 100.0 * correct / N
    return accuracy, x_thresh, y_thresh


def visualize_test_loader_static(model, test_loader, device='cpu', max_samples=300):
    """
    Static scatter plot visualization with lines connecting GT to PD.
    Visualizes only up to 200 samples, but calculates accuracy over all samples.
    """
    frames_data = []  # Each item: { 'x_gt','y_gt','yaw_gt','scan':..., 'x_pd','y_pd','yaw_pd'}

    model.eval()
    gt_list = []
    pd_list = []

    with torch.no_grad():
        for lidar_batch, odom_batch in test_loader:
            B = lidar_batch.size(0)
            for i in range(B):
                single_scan = lidar_batch[i].to(device).unsqueeze(0)  # shape (1,num_beams)
                x_gt, y_gt, yaw_gt = odom_batch[i].tolist()  # ground truth
                pred = model(single_scan)
                x_pd, y_pd, yaw_pd = pred.cpu().numpy()[0]

                frames_data.append({
                    "x_gt": x_gt,
                    "y_gt": y_gt,
                    "yaw_gt": yaw_gt,
                    "scan": lidar_batch[i].cpu().numpy(),
                    "x_pd": x_pd,
                    "y_pd": y_pd,
                    "yaw_pd": yaw_pd
                })

                gt_list.append([x_gt, y_gt, yaw_gt])
                pd_list.append([x_pd, y_pd, yaw_pd])

    # Calculate accuracy over all samples
    gt_array = np.array(gt_list)
    pd_array = np.array(pd_list)
    acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(gt_array, pd_array, 0.1, 0.1)
    setup_logger().info(f"XY Accuracy in test set: {acc_perc:.2f}% (over {len(gt_array)} samples)")

    # Use only up to 200 samples for visualization
    if len(frames_data) > 300:
        visualization_data = frames_data[:300]
    else:
        visualization_data = frames_data

    # Static scatter plot with connection lines
    fig, ax = plt.subplots(figsize=(10, 10))
    legend_labels = set()  # Keep track of added labels

    for data in visualization_data:
        x_gt, y_gt = data["x_gt"], data["y_gt"]
        x_pd, y_pd = data["x_pd"], data["y_pd"]

        # Add scatter points for GT and PD, checking for legend duplication
        if "Ground Truth" not in legend_labels:
            ax.scatter(x_gt, y_gt, color='blue', alpha=0.7, label='Ground Truth')
            legend_labels.add("Ground Truth")
        else:
            ax.scatter(x_gt, y_gt, color='blue', alpha=0.7)

        if "Predicted" not in legend_labels:
            ax.scatter(x_pd, y_pd, color='red', alpha=0.7, label='Predicted')
            legend_labels.add("Predicted")
        else:
            ax.scatter(x_pd, y_pd, color='red', alpha=0.7)

        # Draw a line connecting GT to PD
        ax.plot([x_gt, x_pd], [y_gt, y_pd], color='gray', linestyle='--', alpha=0.5)

    # Annotate accuracy
    acc_text = (f"Accuracy: {acc_perc:.2f}%\n"
                f"total samples: {max_samples}\n"
                f"X Threshold: {x_thresh:.2f} m\n"
                f"Y Threshold: {y_thresh:.2f} m")
    ax.text(0.02, 0.98, acc_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    ax.legend(loc='lower right', framealpha=0.7)
    ax.set_title("Static Visualization: Ground Truth vs Predicted Positions")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)

    plt.show()
    
def visualize_test_loader_static_dicp(model, test_loader, device='cpu', max_samples=300):
    """
    Static scatter plot visualization with lines connecting Ground Truth to Predicted transformations.
    Adapted for LocalizationNet which outputs a transformation [theta, tx, ty] based on 
    inputs (lidar_scan, source_points, target_points). Only up to max_samples are visualized.
    """
    frames_data = []  # Each item: { 'x_gt','y_gt','yaw_gt','scan', 'x_pd','y_pd','yaw_pd'}
    gt_list = []
    pd_list = []
    
    model.eval()
    with torch.no_grad():
        for lidar_batch, odom_batch, source_points, target_points in test_loader:
            B = lidar_batch.size(0)
            for i in range(B):
                # Ensure each input is on the proper device and of float type
                single_scan = lidar_batch[i].to(device).unsqueeze(0)      # shape (1, num_beams)
                sp = source_points[i].to(device).unsqueeze(0)               # shape (1, N, 2)
                tp = target_points[i].to(device).unsqueeze(0)               # shape (1, N, 2)
                
                # Ground truth transformation (e.g., [tx, ty, yaw] or [yaw, tx, ty])
                x_gt, y_gt, yaw_gt = odom_batch[i].tolist()  # adjust order if necessary

                # Pass all three inputs to the model
                pred = model(single_scan, sp, tp)
                pred = torch.stack([pred[:, 1], pred[:, 2], pred[:, 0]], dim=1)
                x_pd, y_pd, yaw_pd = pred.cpu().numpy()[0]
                
                frames_data.append({
                    "x_gt": x_gt,
                    "y_gt": y_gt,
                    "yaw_gt": yaw_gt,
                    "scan": lidar_batch[i].cpu().numpy(),
                    "x_pd": x_pd,
                    "y_pd": y_pd,
                    "yaw_pd": yaw_pd
                })
                
                gt_list.append([x_gt, y_gt, yaw_gt])
                pd_list.append([x_pd, y_pd, yaw_pd])
                
                # Break if we have reached the maximum number of samples to visualize
                if len(frames_data) >= max_samples:
                    break
            if len(frames_data) >= max_samples:
                break

    # Calculate accuracy over all samples using your provided function
    gt_array = np.array(gt_list)
    pd_array = np.array(pd_list)
    acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(gt_array, pd_array, 0.1, 0.1)
    setup_logger().info(f"XY Accuracy in test set: {acc_perc:.2f}% (over {len(gt_array)} samples)")

    # Use the collected visualization data
    visualization_data = frames_data

    # Create the static scatter plot with lines connecting GT to PD
    fig, ax = plt.subplots(figsize=(10, 10))
    legend_labels = set()
    for data in visualization_data:
        x_gt, y_gt = data["x_gt"], data["y_gt"]
        x_pd, y_pd = data["x_pd"], data["y_pd"]

        if "Ground Truth" not in legend_labels:
            ax.scatter(x_gt, y_gt, color='blue', alpha=0.7, label='Ground Truth')
            legend_labels.add("Ground Truth")
        else:
            ax.scatter(x_gt, y_gt, color='blue', alpha=0.7)

        if "Predicted" not in legend_labels:
            ax.scatter(x_pd, y_pd, color='red', alpha=0.7, label='Predicted')
            legend_labels.add("Predicted")
        else:
            ax.scatter(x_pd, y_pd, color='red', alpha=0.7)

        # Draw connection line between ground truth and prediction
        ax.plot([x_gt, x_pd], [y_gt, y_pd], color='gray', linestyle='--', alpha=0.5)

    # Annotate accuracy and sample thresholds
    acc_text = (f"Accuracy: {acc_perc:.2f}%\n"
                f"Total samples: {len(gt_array)}\n"
                f"X Threshold: {x_thresh:.2f} m\n"
                f"Y Threshold: {y_thresh:.2f} m")
    ax.text(0.02, 0.98, acc_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.legend(loc='lower right', framealpha=0.7)
    ax.set_title("Static Visualization: Ground Truth vs Predicted Transformations")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)
    
    plt.show()