import matplotlib.pyplot as plt
import numpy as np
import torch

def calc_accuracy_percentage_xy(gt_array: np.ndarray, pred_array: np.ndarray, x_thresh: float = 0.1, y_thresh: float = 0.1):
    """
    Compares ground truth vs. predicted 2D positions in arrays of shape (N,2).
    Returns the percentage of samples where the absolute differences in both x and y
    are within the given thresholds.
    """
    N = gt_array.shape[0]
    if N == 0:
        return 0.0, x_thresh, y_thresh

    correct = 0
    for i in range(N):
        x_gt, y_gt = gt_array[i]
        x_pd, y_pd = pred_array[i]
        if abs(x_pd - x_gt) <= x_thresh and abs(y_pd - y_gt) <= y_thresh:
            correct += 1
    accuracy = 100.0 * correct / N
    return accuracy, x_thresh, y_thresh

def integrate_velocities(pred_velocities, timestamps, initial_position):
    """
    Integrates a sequence of predicted velocities over time using simple Euler integration.
    Only the first two components (linear velocities in x and y) are used.
    
    :param pred_velocities: Array of shape (N, 3) where columns 0 and 1 are linear velocities.
    :param timestamps: Array of N timestamps (in seconds) sorted in ascending order.
    :param initial_position: Starting position as an array [x0, y0].
    :return: Array of integrated positions of shape (N, 2).
    """
    positions = [np.array(initial_position)]
    for i in range(1, len(pred_velocities)):
        delta_t = timestamps[i] - timestamps[i - 1]
        new_position = positions[-1] + np.array(pred_velocities[i][:2]) * delta_t
        positions.append(new_position)
    return np.array(positions)

def visualize_velocity_predictions(model, test_loader, device='cpu', max_samples=300):
    """
    Visualizes integrated predicted positions versus ground truth positions.
    The test_loader should yield (lidar_input, velocity_target, position_target, timestamp) tuples.
    
    This function:
      - Runs the model on the lidar inputs to predict velocities.
      - Collects predicted velocities, ground truth positions, and timestamps.
      - Sorts the data by timestamp and integrates the predicted velocities to obtain positions.
      - Computes an accuracy metric based on a per-axis threshold.
      - Generates a scatter plot showing ground truth and integrated predicted positions,
        with lines connecting corresponding points.
    """
    model.eval()
    pred_velocities = []
    gt_positions = []
    timestamps = []

    with torch.no_grad():
        for batch in test_loader:
            lidar_input, velocity_target, position_target, timestamp = batch
            lidar_input = lidar_input.to(device)
            preds = model(lidar_input)  # Predicted velocities
            pred_velocities.extend(preds.cpu().numpy())
            gt_positions.extend(position_target.numpy())
            # Handle timestamp conversion if needed
            if isinstance(timestamp, torch.Tensor):
                timestamps.extend(timestamp.cpu().numpy())
            else:
                timestamps.extend(timestamp)

    pred_velocities = np.array(pred_velocities)
    gt_positions = np.array(gt_positions)
    timestamps = np.array(timestamps)

    # Ensure the samples are sorted by timestamp for proper integration.
    sort_idx = np.argsort(timestamps)
    sorted_timestamps = timestamps[sort_idx]
    sorted_pred_velocities = pred_velocities[sort_idx]
    sorted_gt_positions = gt_positions[sort_idx]

    # Integrate the predicted velocities to obtain position estimates.
    initial_position = sorted_gt_positions[0]
    integrated_pred_positions = integrate_velocities(sorted_pred_velocities, sorted_timestamps, initial_position)

    # Calculate the accuracy between the integrated positions and ground truth.
    acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(sorted_gt_positions, integrated_pred_positions)
    print(f"Integrated XY Accuracy: {acc_perc:.2f}% (over {len(sorted_gt_positions)} samples)")

    # Limit visualization to a subset if desired.
    vis_indices = np.arange(min(max_samples, len(sorted_gt_positions)))

    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot ground truth positions.
    ax.scatter(sorted_gt_positions[vis_indices, 0], sorted_gt_positions[vis_indices, 1],
               color='blue', alpha=0.7, label='Ground Truth')
    # Plot integrated predicted positions.
    ax.scatter(integrated_pred_positions[vis_indices, 0], integrated_pred_positions[vis_indices, 1],
               color='red', alpha=0.7, label='Predicted')

    # Draw lines connecting each ground truth to its corresponding prediction.
    for i in vis_indices:
        ax.plot([sorted_gt_positions[i, 0], integrated_pred_positions[i, 0]],
                [sorted_gt_positions[i, 1], integrated_pred_positions[i, 1]],
                color='gray', linestyle='--', alpha=0.5)

    # Annotate the plot with accuracy information.
    acc_text = (f"Integrated Accuracy: {acc_perc:.2f}%\n"
                f"Total Samples: {len(sorted_gt_positions)}\n"
                f"X Threshold: {x_thresh:.2f} m, Y Threshold: {y_thresh:.2f} m")
    ax.text(0.02, 0.98, acc_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    ax.legend(loc='lower right', framealpha=0.7)
    ax.set_title("Visualization: Integrated Predicted vs. Ground Truth Positions")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)

    plt.show()
