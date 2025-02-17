import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt

def polar_to_cartesian(ranges, angle_min=np.pi, angle_max=-np.pi):
    """
    Converts an array of ranges into 2D Cartesian coordinates.
    Assumes angles are evenly spaced from angle_min to angle_max.
    """
    N = len(ranges)
    angles = np.linspace(angle_min, angle_max, N, endpoint=False)
    x = ranges * -np.cos(angles)
    y = ranges * np.sin(angles)
    return np.stack([x, y], axis=1)  # Shape: (N, 2)

def compute_relative_transformation(odom1, odom2):
    """
    Computes the relative transformation (dtheta, tx, ty) from odom1 to odom2.
    odom1 and odom2 are arrays: [pos_x, pos_y, theta]
    The translation is computed in the coordinate frame of odom1.
    """
    x1, y1, theta1 = odom1
    x2, y2, theta2 = odom2
    # Compute relative rotation
    dtheta = theta2 - theta1
    # Normalize angle to [-pi, pi]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    # Compute relative translation in world frame
    dx = x2 - x1
    dy = y2 - y1
    # Rotate into the source frame:
    tx = np.cos(theta1) * dx + np.sin(theta1) * dy
    ty = -np.sin(theta1) * dx + np.cos(theta1) * dy
    return np.array([dtheta, tx, ty], dtype=np.float32)

class LidarOdomPairDataset(Dataset):
    """
    Constructs pairs of consecutive (or gap-separated) scans.
    For each sample index i, returns:
      - lidar_input: raw ranges from sample i (for feature extraction)
      - gt_transformation: relative transformation from sample i to sample i+gap,
                           computed from odometry [pos_x, pos_y, orientation_z]
      - source_points: point cloud from sample i (converted from ranges)
      - target_points: point cloud from sample i+gap (converted from ranges)
    """
    def __init__(self, odom_csv_path: str, scan_csv_path: str, gap=1):
        odom_df = pd.read_csv(odom_csv_path)
        scan_df = pd.read_csv(scan_csv_path)

        # Merge dataframes on [sec, nanosec]
        merged_df = pd.merge(scan_df, odom_df, on=['sec', 'nanosec'], how='inner')
        # Convert 'ranges' (a space-separated string) to a list of floats
        merged_df['ranges_list'] = merged_df['ranges'].apply(
            lambda r: [float(val) for val in r.split(' ')]
        )
        self.data = merged_df
        self.gap = gap

    def __len__(self):
        # Last 'gap' samples cannot form a pair
        return len(self.data) - self.gap

    def __getitem__(self, idx):
        # Source sample is at index idx; target is idx+gap
        row_source = self.data.iloc[idx]
        row_target = self.data.iloc[idx + self.gap]

        # Raw lidar scan as input (source scan)
        lidar_input = np.array(row_source['ranges_list'], dtype=np.float32)

        # Convert raw ranges to 2D point clouds
        source_points = polar_to_cartesian(lidar_input)
        target_points = polar_to_cartesian(np.array(row_target['ranges_list'], dtype=np.float32))

        # Extract odometry: using pos_x, pos_y, orientation_z
        odom_source = np.array([row_source['pos_x'], row_source['pos_y'], row_source['orientation_z']], dtype=np.float32)
        odom_target = np.array([row_target['pos_x'], row_target['pos_y'], row_target['orientation_z']], dtype=np.float32)

        # Compute the ground truth relative transformation from source to target
        gt_transformation = compute_relative_transformation(odom_source, odom_target)

        # Convert outputs to torch tensors
        lidar_input = torch.from_numpy(lidar_input)
        gt_transformation = torch.from_numpy(gt_transformation)
        source_points = torch.from_numpy(source_points)
        target_points = torch.from_numpy(target_points)

        return lidar_input, gt_transformation, source_points, target_points
    
    
def test_dataset_visualization(odom_csv_path, scan_csv_path, gap=1, num_samples=5):
    # Ensure the provided file paths exist.
    if not os.path.exists(odom_csv_path):
        raise FileNotFoundError(f"Odometry file not found: {odom_csv_path}")
    if not os.path.exists(scan_csv_path):
        raise FileNotFoundError(f"Scan file not found: {scan_csv_path}")
    
    # Create the dataset instance.
    dataset = LidarOdomPairDataset(odom_csv_path, scan_csv_path, gap=gap)
    print(f"Total samples in dataset (after pairing): {len(dataset)}\n")
    
    # Loop over a few samples.
    for idx in range(min(num_samples, len(dataset))):
        # Each sample returns: lidar_input, gt_transformation, source_points, target_points.
        lidar_input, gt_transformation, source_points, target_points = dataset[idx]
        
        print(f"Sample {idx}:")
        print(f"  Lidar input shape: {lidar_input.shape} | First 5 values: {lidar_input[:5]}")
        print(f"  Ground Truth Transformation: {gt_transformation}")
        print(f"  Source points shape: {source_points.shape}")
        print(f"  Target points shape: {target_points.shape}\n")
        
        # Visualize the source and target point clouds.
        plt.figure(figsize=(6, 6))
        plt.scatter(source_points[:, 0], source_points[:, 1], c="blue", s=10, label="Source")
        plt.scatter(target_points[:, 0], target_points[:, 1], c="red", s=10, label="Target")
        plt.title(f"Sample {idx}: Source vs Target Point Clouds")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
if __name__ == "__main__":
    # Define paths to your CSV files.
    odom_csv = os.path.join("dataset", "odom_data.csv")
    scan_csv = os.path.join("dataset", "scan_data.csv")
    
    # Visualize a portion of the dataset (e.g. 5 samples).
    test_dataset_visualization(odom_csv, scan_csv, gap=3, num_samples=10)