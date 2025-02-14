from ICP.ICP_utils import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ICP.ICP_utils import *
from ICP.ICP_variants import point_to_point_icp, coarse_to_fine_icp, robust_icp
from dataset import LidarOdomDataset  # Load dataset class
from utils.utils import calc_accuracy_percentage_xy
from utils.metrics import *
import numpy as np

if __name__ == "__main__":
    # Load the merged dataset using LidarOdomDataset
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"
    dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)

    # Create a Lidar object with corrected angle bounds
    lidar = Lidar(np.pi, -np.pi, 360)

    start = 0
    end = 250 #len(dataset)
    gap = 1

    pose = [0, 0, 0]  # [x, y, theta] estimated by ICP
    traj = []       # ICP estimated trajectory (list of poses)
    gt_traj = []    # Ground truth trajectory (from odometry)

    #plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    # Process raw scan: filter and convert from polar to Cartesian coordinates
    def process_scan(scan, usableRange=10):
        scan = np.array(scan)
        valid = (scan > lidar.range_min) & (scan < min(lidar.range_max, usableRange))
        angles = lidar.angles[valid]
        ranges_valid = scan[valid]
        return pol2cart(angles, ranges_valid)

    for i in range(start, end - gap, gap):
        # Get LiDAR scan and odometry data from the dataset
        lidar_input_before, odom_before = dataset[i]
        lidar_input_current, odom_current = dataset[i + gap]

        # Process the raw LiDAR data into Cartesian coordinates
        scan_before_local = process_scan(lidar_input_before)
        scan_current_local = process_scan(lidar_input_current)

        # Transform the scans from the local (sensor) frame to the global frame using the current pose
        scan_before_global = localToGlobal(pose, scan_before_local)
        scan_current_global = localToGlobal(pose, scan_current_local)

        ax.clear()
        ax.set_xlim([-2, 10])
        ax.set_ylim([-2, 10])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        
        # Plot current pose and the ICP estimated trajectory
        traj.append(pose)
        traj_array = np.array(traj)
        # Plot the ICP trajectory (black dotted) and ground truth (green dashed)
        ax.plot(traj_array[:, 0], traj_array[:, 1], color='black', linestyle='dotted', label="ICP Trajectory")
        

        # Apply ICP to align the current scan to the previous scan
        T = point_to_point_icp(scan_current_global, scan_before_global)

        # Update the pose: you might consider smoothing these updates to reduce the wiggle effect
        pose_T = v2t(pose)
        pose = t2v(np.dot(T, pose_T))

        # Transform current scan with T and plot it as red dots (ICP adjusted scan)
        frame = np.ones((3, scan_current_global.shape[0]))
        frame[:2, :] = scan_current_global.T
        result = (T @ frame)[:2, :].T
        ax.plot(result[:, 0], result[:, 1], 'o', markersize=1, color='red', label="ICP Adjusted Scan")

        # Plot ground truth trajectory (using pos_x, pos_y from odometry)
        gt_position = odom_current[:3]
        gt_traj.append(gt_position)
        gt_traj_array = np.array(gt_traj)
        #plt.plot(gt_traj_array[:, 0], gt_traj_array[:, 1], color='green', linestyle='--', label="Ground Truth")
        ax.plot(gt_traj_array[:, 0], gt_traj_array[:, 1], color='green', linestyle='--', label="Ground Truth")
        # Calculate mean error between estimated and ground truth positions
        if len(traj_array) == len(gt_traj_array):
            errors = np.linalg.norm(traj_array - gt_traj_array, axis=1)
            mean_error = np.mean(errors)
            rmse = compute_rmse(gt_traj_array, traj_array)
            l1 = compute_mae_L1(gt_traj_array, traj_array)
            STD = compute_std_error(gt_traj_array, traj_array)
        else:
            mean_error = 0.0

        # Compute accuracy percentage (samples within threshold error)
        accuracy, _ , _ = calc_accuracy_percentage_xy(gt_traj_array, traj_array)

        # Place text in upper-left corner (axes fraction coordinates):
        # x=0.02 means 2% from left, y=0.98 means 2% from top
        ax.text(0.02, 0.98,
                f"RMSE: {rmse:.4f} m\n"
                f"Mean = L2 = Euclidean  : {mean_error:.4f} m\n"
                f"Mean = L1 = Manhattan: {l1:.4f} m\n"
                f"standart deviation: {STD:.4f} m\n"
                f"Accuracy: {accuracy:.2f}%",
                fontsize=8, color='black',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))  # align text block at top

        ax.legend(loc='upper right', fontsize=8)
        #plt.draw()
        plt.pause(0.07)
    # num_frames = (end - start) // gap
    # ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
    # Save as GIF (requires Pillow)
    # ani.save("icp_animation.gif", writer="pillow", fps=10)
    # Save as MP4 (requires ffmpeg)
    # ani.save("icp_animation.mp4", writer="ffmpeg", fps=10)
    plt.show()
