import numpy as np
import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Setup Python logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
from diffusion import forward_diffusion

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
        x_gt, y_gt = gt_array[i]
        x_pd, y_pd = pred_array[i]
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
    T = 50
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    with torch.no_grad():
        for lidar_batch, odom_batch in test_loader:
            lidar_batch = lidar_batch.to(device)  # Move `lidar_batch` to device
            odom_batch = odom_batch.to(device)  # Move `odom_batch` to device
            B = lidar_batch.size(0)
            for i in range(B):
                single_scan = lidar_batch[i].unsqueeze(0)  # shape (1,num_beams)
                x_gt, y_gt, yaw_gt = odom_batch[i].tolist()  # ground truth
                t = torch.randint(0, T, (1,), device=device)
                #pose_t, eps = forward_diffusion(odom_batch, t, betas)
                pose_t, eps = forward_diffusion(odom_batch[i].unsqueeze(0), t, betas)
                pred_noise  = model(pose_t, single_scan, t)

                x_pd, y_pd, yaw_pd = pred_noise.cpu().numpy()[0]

                frames_data.append({
                    "x_gt": x_gt,
                    "y_gt": y_gt,
                    "yaw_gt": yaw_gt,
                    "scan": lidar_batch[i].cpu().numpy(),
                    "x_pd": x_pd,
                    "y_pd": y_pd,
                    "yaw_pd": yaw_pd
                })

                gt_list.append([x_gt, y_gt])
                pd_list.append([x_pd, y_pd])

    # Calculate accuracy over all samples
    gt_array = np.array(gt_list)
    pd_array = np.array(pd_list)
    acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(gt_array, pd_array, 0.1, 0.1)
    logger.info(f"XY Accuracy in test set: {acc_perc:.2f}% (over {len(gt_array)} samples)")

    # Use only up to 200 samples for visualization
    if len(frames_data) > 200:
        visualization_data = frames_data[:200]
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