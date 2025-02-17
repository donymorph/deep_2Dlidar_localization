import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class LidarOdomDataset(Dataset):
    """
    Loads LiDAR and odometry CSV files, merges them on [sec, nanosec],
    and provides (lidar_input, velocity_target, position_target, timestamp) tuples.
    """
    def __init__(self, odom_csv_path: str, scan_csv_path: str):
        odom_df = pd.read_csv(odom_csv_path)
        scan_df = pd.read_csv(scan_csv_path)

        # Merge on sec, nanosec
        merged_df = pd.merge(scan_df, odom_df, on=['sec','nanosec'], how='inner')

        # Convert 'ranges' from space-separated string to list of floats
        merged_df['ranges_list'] = merged_df['ranges'].apply(
            lambda r: [float(val) for val in r.split(' ')]
        )
        # Compute a floating point timestamp (sec + nanosec*1e-9)
        merged_df['timestamp'] = merged_df['sec'] + merged_df['nanosec'] * 1e-9

        self.data = merged_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # LiDAR input remains unchanged
        lidar_input = np.array(row['ranges_list'], dtype=np.float32)
        # Use velocity data as targets (linear velocities and angular velocity)
        velocity_target = np.array([
            row['linear_x'],
            row['linear_y'],
            row['angular_z']
        ], dtype=np.float32)
        # Ground truth positions are kept for integration evaluation
        position_target = np.array([
            row['pos_x'],
            row['pos_y']
        ], dtype=np.float32)
        timestamp = row['timestamp']
        return lidar_input, velocity_target, position_target, timestamp
    
def integrate_velocities(pred_velocities, timestamps, initial_position):
    """
    Integrates a sequence of predicted velocities (assumed in global frame) over time.
    Only the first two components (linear velocities) are used for position integration.
    :param pred_velocities: Array of shape (N, 3), where columns 0 and 1 are linear velocities.
    :param timestamps: Array of N timestamps (in seconds) sorted in ascending order.
    :param initial_position: Starting position as an array [x0, y0].
    :return: Array of integrated positions of shape (N, 2).
    """
    positions = [initial_position]
    for i in range(1, len(pred_velocities)):
        delta_t = timestamps[i] - timestamps[i - 1]
        # For simplicity, assume the predicted linear velocities are already in the global frame.
        new_position = positions[-1] + np.array(pred_velocities[i][:2]) * delta_t
        positions.append(new_position)
    return np.array(positions)