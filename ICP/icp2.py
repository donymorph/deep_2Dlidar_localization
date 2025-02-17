import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ICP.ICP_utils import Lidar, pol2cart, v2t, t2v, localToGlobal
import matplotlib.pyplot as plt
import numpy as np
from ICP.ICP_variants import (
    point_to_point_icp, point_to_plane_icp, 
    generalized_icp, coarse_to_fine_icp, 
    robust_icp, multi_scale_icp,
    sparse_icp, symmetric_icp)
from dataset import LidarOdomDataset
from utils.utils import calc_accuracy_percentage_xy
from utils.metrics import compute_rmse, compute_mae_L1, compute_std_error

if __name__ == "__main__":
    # Load dataset
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"
    dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)

    # Ensure dataset is not empty
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the CSV files.")

    # Create LiDAR object
    lidar = Lidar(np.pi, -np.pi, 360)

    start = 0
    end = min(300, len(dataset))  # Ensure we don't exceed dataset length
    gap = 1

    # **Initialize pose from dataset's first entry**
    first_entry = dataset[0][1]  # Extract odometry data from first entry
    pose = [first_entry[0], first_entry[1], first_entry[2]]
    #pose = [0, 0, 0]  # [x, y, theta]
    traj = []       # Stores estimated trajectory from ICP
    gt_traj = []    # Stores ground truth trajectory from odometry

    fig, ax = plt.subplots(figsize=(8, 6))

    def process_scan(scan, usable_range=10):
        """Process raw LiDAR scan to filter valid points and convert to Cartesian."""
        scan = np.array(scan)
        valid = (scan > lidar.range_min) & (scan < min(lidar.range_max, usable_range))
        angles = lidar.angles[valid]
        ranges_valid = scan[valid]
        return pol2cart(angles, ranges_valid)

    for i in range(start, end - gap, gap):
        # **Get LiDAR scan and odometry data**
        lidar_input_before, odom_before = dataset[i]
        lidar_input_current, odom_current = dataset[i + gap]

        # **Process LiDAR scans**
        scan_before_local = process_scan(lidar_input_before)
        scan_current_local = process_scan(lidar_input_current)

        # **Transform scans from sensor frame to global frame**
        scan_before_global = localToGlobal(pose, scan_before_local)
        scan_current_global = localToGlobal(pose, scan_current_local)

        ax.clear()
        ax.set_xlim([-2, 10])
        ax.set_ylim([-2, 10])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        # **Plot the trajectory**
        traj.append(pose)
        traj_array = np.array(traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1], color='black', linestyle='dotted', label="ICP Trajectory")

        # **Apply ICP**
        T = coarse_to_fine_icp(scan_current_global, scan_before_global)

        # **Update estimated pose**
        pose_T = v2t(pose)
        pose = t2v(np.dot(T, pose_T))

        # **Transform and plot ICP adjusted scan**
        frame = np.ones((3, scan_current_global.shape[0]))
        frame[:2, :] = scan_current_global.T
        result = (T @ frame)[:2, :].T
        ax.plot(result[:, 0], result[:, 1], 'o', markersize=1, color='red', label="ICP Adjusted Scan")

        # **Plot ground truth trajectory**
        gt_position = [odom_current[0], odom_current[1], odom_current[2]]
        gt_traj.append(gt_position)
        gt_traj_array = np.array(gt_traj)
        ax.plot(gt_traj_array[:, 0], gt_traj_array[:, 1], color='green', linestyle='--', label="Ground Truth")

        # **Compute errors and accuracy**
        if len(traj_array) == len(gt_traj_array):
            errors = np.linalg.norm(traj_array - gt_traj_array, axis=1)
            mean_error = np.mean(errors)
            rmse = compute_rmse(gt_traj_array, traj_array)
            l1 = compute_mae_L1(gt_traj_array, traj_array)
            std_dev = compute_std_error(gt_traj_array, traj_array)
        else:
            mean_error, rmse, l1, std_dev = 0.0, 0.0, 0.0, 0.0

        # **Compute accuracy percentage**
        accuracy, _, _ = calc_accuracy_percentage_xy(gt_traj_array, traj_array)

        # **Display statistics**
        ax.text(0.02, 0.98,
                f"RMSE: {rmse:.4f} m\n"
                f"Mean Error: {mean_error:.4f} m\n"
                f"Manhattan Distance: {l1:.4f} m\n"
                f"Standard Deviation: {std_dev:.4f} m\n"
                f"Accuracy: {accuracy:.2f}%",
                fontsize=8, color='black',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))

        ax.legend(loc='upper right', fontsize=8)
        plt.pause(0.07)

    plt.show()
