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
