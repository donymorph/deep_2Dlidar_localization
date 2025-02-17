# dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class LidarOdomDataset(Dataset):
    """
    Loads LiDAR and odometry CSV files, merges them on [sec, nanosec],
    and provides (lidar_input, odom_output) pairs.
    """
    def __init__(self, odom_csv_path: str, scan_csv_path: str):
        odom_df = pd.read_csv(odom_csv_path)
        scan_df = pd.read_csv(scan_csv_path)

        # Merge on sec,nanosec
        merged_df = pd.merge(scan_df, odom_df, on=['sec','nanosec'], how='inner')

        # Convert 'ranges' from space-separated string to list of floats
        merged_df['ranges_list'] = merged_df['ranges'].apply(
            lambda r: [float(val) for val in r.split(' ')]
        )
        # merged_df['ranges_list'] = merged_df['ranges_list'].apply(
        #     lambda arr: [val / 10.0 for val in arr]  # scale [0..10] down to [0..1]
        # )

        # Store the entire merged dataframe
        self.data = merged_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # LiDAR input
        lidar_input = np.array(row['ranges_list'], dtype=np.float32)

        # Odom output (position, orientatin, linear_vel, angular_vel)
        # depends on what you want to predict you can comment out
        odom_output = np.array([
            row['pos_x'],
            row['pos_y'],
            #row['pos_z'],
            #row['orientation_x'],
            #row['orientation_y'],
            row['orientation_z'],
            #row['linear_x'],
            #row['linear_y'],
            #row['linear_z'],
            #row['angular_x'],
            #row['angular_y'],
            #row['angular_z'],
        ], dtype=np.float32)

        return lidar_input, odom_output

class LidarOdomDataset_Tyler(Dataset):
    def __init__(self, odom_csv_path: str, scan_csv_path: str):
        odom_df = pd.read_csv(odom_csv_path)
        scan_df = pd.read_csv(scan_csv_path)
        
        # Merge datasets
        merged_df = pd.merge(scan_df, odom_df, on=['sec','nanosec'], how='inner')
        
        # Convert ranges to float arrays and normalize
        merged_df['ranges_list'] = merged_df['ranges'].apply(
            lambda r: [float(val) for val in r.split(' ')]
        )
        
        # Calculate normalization factors
        ranges_arrays = np.vstack(merged_df['ranges_list'].values)
        self.ranges_mean = np.mean(ranges_arrays)
        self.ranges_std = np.std(ranges_arrays)
        
        # Store position normalization factors
        self.pos_x_mean = odom_df['pos_x'].mean()
        self.pos_x_std = odom_df['pos_x'].std()
        self.pos_y_mean = odom_df['pos_y'].mean() 
        self.pos_y_std = odom_df['pos_y'].std()
        self.orientation_z_mean = odom_df['orientation_z'].mean()
        self.orientation_z_std = odom_df['orientation_z'].std()
        
        self.data = merged_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Normalize LiDAR input
        lidar_input = np.array(row['ranges_list'], dtype=np.float32)
        lidar_input = (lidar_input - self.ranges_mean) / self.ranges_std
        
        # Normalize odometry output
        pos_x = (row['pos_x'] - self.pos_x_mean) / self.pos_x_std
        pos_y = (row['pos_y'] - self.pos_y_mean) / self.pos_y_std
        orientation_z = (row['orientation_z'] - self.orientation_z_mean) / self.orientation_z_std
        
        odom_output = np.array([pos_x, pos_y, orientation_z], dtype=np.float32)
        
        return lidar_input, odom_output
        
    def denormalize_output(self, normalized_output):
        """Denormalize model predictions back to real values."""
        pos_x = normalized_output[0] * self.pos_x_std + self.pos_x_mean
        pos_y = normalized_output[1] * self.pos_y_std + self.pos_y_mean
        orientation_z = normalized_output[2] * self.orientation_z_std + self.orientation_z_mean
        return np.array([pos_x, pos_y, orientation_z])
    
class LidarOdomDataset_withNoise(Dataset):
    def __init__(self, odom_csv_path: str, scan_csv_path: str, lidar_noise_std=0.01, odom_noise_std=0.01):
        """
        Args:
            odom_csv_path (str): Path to the odometry CSV file.
            scan_csv_path (str): Path to the scan CSV file.
            lidar_noise_std (float): Standard deviation for Gaussian noise in LiDAR ranges.
            odom_noise_std (float): Standard deviation for Gaussian noise in odometry outputs.
        """
        odom_df = pd.read_csv(odom_csv_path)
        scan_df = pd.read_csv(scan_csv_path)
        
        # Merge datasets
        merged_df = pd.merge(scan_df, odom_df, on=['sec', 'nanosec'], how='inner')
        
        # Convert ranges to float arrays
        merged_df['ranges_list'] = merged_df['ranges'].apply(
            lambda r: [float(val) for val in r.split(' ')]
        )
        
        # Calculate normalization factors
        ranges_arrays = np.vstack(merged_df['ranges_list'].values)
        self.ranges_mean = np.mean(ranges_arrays)
        self.ranges_std = np.std(ranges_arrays)
        
        # Store position normalization factors
        self.pos_x_mean = odom_df['pos_x'].mean()
        self.pos_x_std = odom_df['pos_x'].std()
        self.pos_y_mean = odom_df['pos_y'].mean()
        self.pos_y_std = odom_df['pos_y'].std()
        self.orientation_z_mean = odom_df['orientation_z'].mean()
        self.orientation_z_std = odom_df['orientation_z'].std()
        
        self.data = merged_df

        # Noise parameters
        self.lidar_noise_std = lidar_noise_std
        self.odom_noise_std = odom_noise_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Normalize LiDAR input and add noise
        lidar_input = np.array(row['ranges_list'], dtype=np.float32)
        lidar_input = (lidar_input - self.ranges_mean) / self.ranges_std
        lidar_noise = np.random.normal(0, self.lidar_noise_std, size=lidar_input.shape)
        lidar_input += lidar_noise  # Add Gaussian noise

        # Normalize odometry output and add noise
        pos_x = (row['pos_x'] - self.pos_x_mean) / self.pos_x_std
        pos_y = (row['pos_y'] - self.pos_y_mean) / self.pos_y_std
        orientation_z = (row['orientation_z'] - self.orientation_z_mean) / self.orientation_z_std
        
        odom_output = np.array([pos_x, pos_y, orientation_z], dtype=np.float32)
        odom_noise = np.random.normal(0, self.odom_noise_std, size=odom_output.shape)
        odom_output += odom_noise  # Add Gaussian noise
        
        return lidar_input, odom_output
        
    def denormalize_output(self, normalized_output):
        """Denormalize model predictions back to real values."""
        pos_x = normalized_output[0] * self.pos_x_std + self.pos_x_mean
        pos_y = normalized_output[1] * self.pos_y_std + self.pos_y_mean
        orientation_z = normalized_output[2] * self.orientation_z_std + self.orientation_z_mean
        return np.array([pos_x, pos_y, orientation_z])

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
# ---------------------------
# Test
# ---------------------------    
import logging
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

def test_lidar_odom_dataset():
    """
    Test function for LidarOdomDataset class.
    """
    logger.info("Starting test for LidarOdomDataset...")
    
    # Paths to CSV files (replace with your paths)
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"

    # Check if files exist
    if not os.path.exists(odom_csv_path) or not os.path.exists(scan_csv_path):
        logger.error("CSV files for testing not found. Please provide valid paths.")
        return

    try:
        # Initialize dataset
        dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)
        logger.info(f"Dataset successfully loaded with {len(dataset)} samples.")

        # Retrieve a sample
        lidar_input, odom_output = dataset[0]
        logger.info(f"Sample LiDAR input shape: {lidar_input.shape}")
        logger.info(f"Sample odometry output: {odom_output}")
    except Exception as e:
        logger.error(f"An error occurred during testing of LidarOdomDataset: {e}")
    else:
        logger.info("Test for LidarOdomDataset completed successfully.")


def test_lidar_odom_dataset_tyler():
    """
    Test function for LidarOdomDataset_Tyler class.
    """
    logger.info("Starting test for LidarOdomDataset_Tyler...")
    
    # Paths to CSV files (replace with your paths)
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"

    # Check if files exist
    if not os.path.exists(odom_csv_path) or not os.path.exists(scan_csv_path):
        logger.error("CSV files for testing not found. Please provide valid paths.")
        return

    try:
        # Initialize dataset
        dataset = LidarOdomDataset_Tyler(odom_csv_path, scan_csv_path)
        logger.info(f"Dataset successfully loaded with {len(dataset)} samples.")

        # Retrieve a sample
        lidar_input, odom_output = dataset[0]
        logger.info(f"Sample LiDAR input shape: {lidar_input.shape}")
        logger.info(f"Sample normalized odometry output: {odom_output}")

        # Test denormalization
        denormalized_output = dataset.denormalize_output(odom_output)
        logger.info(f"Denormalized odometry output: {denormalized_output}")
    except Exception as e:
        logger.error(f"An error occurred during testing of LidarOdomDataset_Tyler: {e}")
    else:
        logger.info("Test for LidarOdomDataset_Tyler completed successfully.")

def visualize_three_datasets_2d(odom_csv_path, scan_csv_path, num_samples=30):
    """
    Loads a portion of data from three different dataset classes and
    visualizes their odometry outputs (pos_x, pos_y, orientation_z)
    in three side-by-side subplots.

    Args:
        odom_csv_path (str): Path to the odometry CSV file.
        scan_csv_path (str): Path to the scan CSV file.
        num_samples (int): Number of samples to visualize from each dataset.
    """
    # 1. Import your dataset classes
    from dataset import (
        LidarOdomDataset,
        LidarOdomDataset_Tyler,
        LidarOdomDataset_withNoise
    )

    # 2. Instantiate each dataset
    dataset_raw = LidarOdomDataset(odom_csv_path, scan_csv_path)
    dataset_normalized = LidarOdomDataset_Tyler(odom_csv_path, scan_csv_path)
    dataset_noisy = LidarOdomDataset_withNoise(odom_csv_path, scan_csv_path,
                                               lidar_noise_std=0.01,
                                               odom_noise_std=0.01)

    # 3. Prepare lists to hold the odometry data
    #    We'll store pos_x, pos_y, orientation_z separately for clarity
    pos_x_raw, pos_y_raw, ori_z_raw = [], [], []
    pos_x_norm, pos_y_norm, ori_z_norm = [], [], []
    pos_x_noisy, pos_y_noisy, ori_z_noisy = [], [], []

    # 4. Grab the first `num_samples` from each dataset (or fewer if the dataset is smaller)
    n_raw = min(num_samples, len(dataset_raw))
    n_norm = min(num_samples, len(dataset_normalized))
    n_noisy = min(num_samples, len(dataset_noisy))

    for i in range(n_raw):
        _, odom_raw = dataset_raw[i]
        pos_x_raw.append(odom_raw[0])
        pos_y_raw.append(odom_raw[1])
        ori_z_raw.append(odom_raw[2])

    for i in range(n_norm):
        _, odom_norm = dataset_normalized[i]
        pos_x_norm.append(odom_norm[0])
        pos_y_norm.append(odom_norm[1])
        ori_z_norm.append(odom_norm[2])

    for i in range(n_noisy):
        _, odom_noisy = dataset_noisy[i]
        pos_x_noisy.append(odom_noisy[0])
        pos_y_noisy.append(odom_noisy[1])
        ori_z_noisy.append(odom_noisy[2])

    # 5. Plot each dataset’s pos_x, pos_y, orientation_z in separate subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # -----------------------------------------------------------------------
    # Subplot 1: LidarOdomDataset (Raw)
    # -----------------------------------------------------------------------
    axes[0].plot(pos_x_raw, label='pos_x')
    axes[0].plot(pos_y_raw, label='pos_y')
    axes[0].plot(ori_z_raw, label='orientation_z')
    axes[0].set_title('LidarOdomDataset (Raw)')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    # -----------------------------------------------------------------------
    # Subplot 2: LidarOdomDataset_Tyler (Normalized)
    # -----------------------------------------------------------------------
    axes[1].plot(pos_x_norm, label='pos_x')
    axes[1].plot(pos_y_norm, label='pos_y')
    axes[1].plot(ori_z_norm, label='orientation_z')
    axes[1].set_title('LidarOdomDataset_Tyler (Normalized)')
    axes[1].set_xlabel('Sample Index')
    axes[1].legend()

    # -----------------------------------------------------------------------
    # Subplot 3: LidarOdomDataset_withNoise (Normalized + Noise)
    # -----------------------------------------------------------------------
    axes[2].plot(pos_x_noisy, label='pos_x')
    axes[2].plot(pos_y_noisy, label='pos_y')
    axes[2].plot(ori_z_noisy, label='orientation_z')
    axes[2].set_title('LidarOdomDataset_withNoise (Normalized+Noise)')
    axes[2].set_xlabel('Sample Index')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def visualize_three_datasets(odom_csv_path, scan_csv_path, num_samples=30):
    """
    Loads a portion of data from three different dataset classes and
    visualizes their odometry outputs (pos_x, pos_y, orientation_z)
    using scatter plots and 3D visualization.

    Args:
        odom_csv_path (str): Path to the odometry CSV file.
        scan_csv_path (str): Path to the scan CSV file.
        num_samples (int): Number of samples to visualize from each dataset.
    """
    # Import dataset classes
    from dataset import (
        LidarOdomDataset,
        LidarOdomDataset_Tyler,
        LidarOdomDataset_withNoise
    )

    # Instantiate each dataset
    dataset_raw = LidarOdomDataset(odom_csv_path, scan_csv_path)
    dataset_normalized = LidarOdomDataset_Tyler(odom_csv_path, scan_csv_path)
    dataset_noisy = LidarOdomDataset_withNoise(odom_csv_path, scan_csv_path,
                                               lidar_noise_std=0.01,
                                               odom_noise_std=0.01)

    # Prepare lists to hold odometry data
    datasets = {
        "Raw Data": {"x": [], "y": [], "z": []},
        "Normalized": {"x": [], "y": [], "z": []},
        "Normalized + Noise": {"x": [], "y": [], "z": []}
    }

    # Function to extract dataset samples
    def extract_samples(dataset, key, num_samples):
        n = min(num_samples, len(dataset))
        for i in range(n):
            _, odom = dataset[i]
            datasets[key]["x"].append(odom[0])  # pos_x
            datasets[key]["y"].append(odom[1])  # pos_y
            datasets[key]["z"].append(odom[2])  # orientation_z

    # Extract data
    extract_samples(dataset_raw, "Raw Data", num_samples)
    extract_samples(dataset_normalized, "Normalized", num_samples)
    extract_samples(dataset_noisy, "Normalized + Noise", num_samples)

    # ---------------------------------------------------------------------
    # **1️⃣ Plot 2D Scatter: pos_x vs pos_y**
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, data in datasets.items():
        ax.scatter(data["x"], data["y"], label=key, alpha=0.7)

    ax.set_title("2D Scatter: pos_x vs pos_y")
    ax.set_xlabel("pos_x")
    ax.set_ylabel("pos_y")
    ax.legend()
    ax.grid()

    # ---------------------------------------------------------------------
    # **2️⃣ 3D Scatter: pos_x, pos_y, orientation_z**
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d")

    for key, data in datasets.items():
        ax3d.scatter(data["x"], data["y"], data["z"], label=key, alpha=0.7)

    ax3d.set_title("3D Scatter: pos_x, pos_y, orientation_z")
    ax3d.set_xlabel("pos_x")
    ax3d.set_ylabel("pos_y")
    ax3d.set_zlabel("orientation_z")
    ax3d.legend()

    # ---------------------------------------------------------------------
    # **3️⃣ Line Plot: Orientation_Z Over Samples**
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, data in datasets.items():
        ax.plot(data["z"], label=key)

    ax.set_title("Orientation Z Comparison")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Orientation Z")
    ax.legend()
    ax.grid()

    # Show all plots
    plt.show()
if __name__ == "__main__":
    # test_lidar_odom_dataset()
    # test_lidar_odom_dataset_tyler()
    # Update these paths to point to your CSV files
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"

    # Visualize
    visualize_three_datasets(odom_csv_path, scan_csv_path, num_samples=200)
    #visualize_three_datasets_2d(odom_csv_path, scan_csv_path, num_samples=200)