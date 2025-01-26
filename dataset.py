# dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class LidarOdomDataset(Dataset):
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