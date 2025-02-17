# dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

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

    def save_odom_data(self, out_csv_filepath:str):
        
        # Select the columns you want to save
        selected_columns = ['pos_x', 'pos_y', 'orientation_z']
        new_df = self.data[selected_columns]
        # Save the selected columns to a CSV file
        new_df.to_csv(out_csv_filepath, index=False)


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
    
import logging
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

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


if __name__ == "__main__":
    test_lidar_odom_dataset()
    test_lidar_odom_dataset_tyler()
