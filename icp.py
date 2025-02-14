import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ICP.ICP_utils import Lidar, pol2cart, v2t, t2v, localToGlobal
from ICP.ICP_variants import (
    point_to_point_icp, point_to_plane_icp, 
    trimmed_icp, robust_icp, sparse_icp,
    multi_scale_icp, symmetric_icp, coarse_to_fine_icp)
from dataset import LidarOdomDataset
from utils.utils import calc_accuracy_percentage_xy
from utils.metrics import compute_rmse, compute_mae_L1, compute_std_error

# Load dataset
odom_csv_path = "dataset/odom_data.csv"
scan_csv_path = "dataset/scan_data.csv"
dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)

# Lidar settings
lidar = Lidar(np.pi, -np.pi, 360)

# Set animation parameters
start = 0
end = 300
gap = 1

pose = [0, 0, 0]  # [x, y, theta]
traj = []
gt_traj = []

fig, ax = plt.subplots(figsize=(8, 6))

# Function to process LiDAR scans
def process_scan(scan, usableRange=10):
    scan = np.array(scan)
    valid = (scan > lidar.range_min) & (scan < min(lidar.range_max, usableRange))
    angles = lidar.angles[valid]
    ranges_valid = scan[valid]
    return pol2cart(angles, ranges_valid)

def compute_normals(points):
    """Compute normals for 2D point cloud."""
    if len(points) < 3:  # If not enough points, return zeros
        return np.zeros_like(points)

    normals = np.zeros_like(points)
    n = len(points)

    for i in range(1, n - 1):
        # Compute the tangent vector using neighbors
        tangent = points[i + 1] - points[i - 1]
        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular vector

        # Normalize the normal
        norm = np.linalg.norm(normal)
        if norm != 0:
            normal /= norm

        normals[i] = normal

    # Handle boundary cases (copy nearest neighbor's normal)
    normals[0] = normals[1]
    normals[-1] = normals[-2]

    return normals
        
# Animation update function
def update(frame_idx):
    global pose  # Ensure pose updates persist across frames
    ax.clear()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(True)

    i = start + frame_idx * gap
    if i >= end - gap:
        return

    # Retrieve LiDAR scans and odometry
    lidar_input_before, odom_before = dataset[i]
    lidar_input_current, odom_current = dataset[i + gap]

    # Process the scans
    scan_before_local = process_scan(lidar_input_before)
    scan_current_local = process_scan(lidar_input_current)

    # Transform to global coordinates
    scan_before_global = localToGlobal(pose, scan_before_local)
    scan_current_global = localToGlobal(pose, scan_current_local)

    # Record trajectory
    traj.append(pose.copy())
    traj_array = np.array(traj)

    # Ground truth trajectory
    gt_position = odom_current[:3]
    gt_traj.append(gt_position)
    gt_traj_array = np.array(gt_traj)

    # Apply ICP
        # Compute normals for point-to-plane ICP
    normals = compute_normals(scan_before_global)
    T = point_to_plane_icp(scan_current_global, scan_before_global, normals)

    # Update pose
    pose_T = v2t(pose)
    pose = t2v(np.dot(T, pose_T))

    # Transform current scan using ICP transformation
    homogeneous_scan = np.ones((3, scan_current_global.shape[0]))  # Create homogeneous coordinates
    homogeneous_scan[:2, :] = scan_current_global.T
    result = (T @ homogeneous_scan)[:2, :].T  # Apply transformation

    # Determine dynamic plot limits to keep all points visible
    all_x = np.concatenate((traj_array[:, 0], gt_traj_array[:, 0], result[:, 0]))
    all_y = np.concatenate((traj_array[:, 1], gt_traj_array[:, 1], result[:, 1]))
    ax.set_xlim([np.min(all_x) - 1, np.max(all_x) + 1])
    ax.set_ylim([np.min(all_y) - 1, np.max(all_y) + 1])

    # Plot ICP and ground truth trajectories
    ax.plot(traj_array[:, 0], traj_array[:, 1], color='green', linestyle='dashed', label="ICP Trajectory")
    ax.plot(gt_traj_array[:, 0], gt_traj_array[:, 1], color='black', linestyle='solid', label="Ground Truth")

    # Plot ICP-adjusted scan
    ax.plot(result[:, 0], result[:, 1], 'o', markersize=1, color='red', label="ICP Adjusted Scan")

    # Compute errors
    if len(traj_array) == len(gt_traj_array):
        errors = np.linalg.norm(traj_array - gt_traj_array, axis=1)
        mean_error = np.mean(errors)
        rmse = compute_rmse(gt_traj_array, traj_array)
        l1 = compute_mae_L1(gt_traj_array, traj_array)
        STD = compute_std_error(gt_traj_array, traj_array)
    else:
        mean_error = 0.0
        rmse = 0.0
        l1 = 0.0
        STD = 0.0

    # Compute accuracy
    accuracy, threshold_x, threshold_y = calc_accuracy_percentage_xy(gt_traj_array, traj_array)

    # Display text information
    ax.text(0.02, 0.98,
            f"RMSE: {rmse:.4f} m\n"
            f"Mean = L2 = Euclidean  : {mean_error:.4f} m\n"
            f"Mean = L1 = Manhattan: {l1:.4f} m\n"
            f"Standard Deviation: {STD:.4f} m\n"
            f"Accuracy: {accuracy:.2f}%",
            fontsize=8, color='black',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))

    ax.text(0.48, 0.98,
            f"Frame: {frame_idx}/{end//gap}\n"
            f"Threshold X: {threshold_x:.2f} m\n"
            f"Threshold Y: {threshold_y:.2f} m\n",
            fontsize=8, color='black',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))

    ax.legend(loc='upper right', fontsize=8)

# Create animation
num_frames = (end - start) // gap
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

# Save as MP4 (requires ffmpeg)
#ani.save("icp_animation.mp4", writer="ffmpeg", fps=10)

# Save as GIF (requires Pillow)
#ani.save("icp_animation.gif", writer="pillow", fps=10)

plt.show()
