# dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class LidarOdomDataset(Dataset):
    """
    Loads LiDAR and odometry CSV files, merges them on [sec, nanosec],
    and provides (lidar_input, odom_output) pairs.
    """
    def __init__(self, odom_csv_path: str, scan_csv_path: str):
        odom_df = pd.read_csv(odom_csv_path)
        scan_df = pd.read_csv(scan_csv_path)

        # Merge on sec, nanosec
        merged_df = pd.merge(scan_df, odom_df, on=['sec', 'nanosec'], how='inner')

        # Convert 'ranges' from space-separated string to list of floats
        merged_df['ranges_list'] = merged_df['ranges'].apply(
            lambda r: [float(val) for val in r.split(' ')]
        )

        # Store the entire merged dataframe
        self.data = merged_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # LiDAR input
        lidar_input = np.array(row['ranges_list'], dtype=np.float32)

        # Odom output (position, orientation, linear_vel, angular_vel)
        odom_output = np.array([
            row['pos_x'],
            row['pos_y'],
            row['pos_z'],
            row['orientation_x'],
            row['orientation_y'],
            row['orientation_z'],
            row['linear_x'],
            row['linear_y'],
            row['linear_z'],
            row['angular_x'],
            row['angular_y'],
            row['angular_z'],
        ], dtype=np.float32)

        return lidar_input, odom_output

def visualize_data(dataset, num_samples=100):
    """
    Visualizes LiDAR data, position, linear velocity, and angular velocity.

    Parameters:
        dataset (LidarOdomDataset): The dataset to visualize.
        num_samples (int): Number of samples to display.
    """
    num_samples = min(num_samples, len(dataset.data))

    # Extract relevant data
    lidar_data = np.stack(dataset.data['ranges_list'].values[:num_samples])
    pos_x = dataset.data['pos_x'].values[:num_samples]
    pos_y = dataset.data['pos_y'].values[:num_samples]
    pos_z = dataset.data['pos_z'].values[:num_samples]

    linear_x = dataset.data['linear_x'].values[:num_samples]
    linear_y = dataset.data['linear_y'].values[:num_samples]
    linear_z = dataset.data['linear_z'].values[:num_samples]

    angular_x = dataset.data['angular_x'].values[:num_samples]
    angular_y = dataset.data['angular_y'].values[:num_samples]
    angular_z = dataset.data['angular_z'].values[:num_samples]

    theta = np.linspace(0, 2 * np.pi, lidar_data.shape[1])

    # Create figure and gridspec layout
    fig = plt.figure(figsize=(14, 7))
    grid = plt.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[1, 1], wspace=0.5, hspace=0.3)

    # 3D Plot: LiDAR Points & Position (Left Side, Bigger)
    ax1 = fig.add_subplot(grid[:, 0], projection='3d')
    for i in range(num_samples):
        x = lidar_data[i] * np.cos(theta)
        y = lidar_data[i] * np.sin(theta)
        z = np.full_like(x, pos_z[i])
        ax1.plot(x, y, z, alpha=0.5)
    ax1.scatter(pos_x, pos_y, pos_z, c=pos_z, cmap='coolwarm', edgecolor='k', s=30)
    ax1.set_title('LiDAR Points & Position')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 2D Plot: Linear Velocity (Top Right)
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.plot(linear_x, label='Linear X', color='blue')
    ax2.plot(linear_y, label='Linear Y', color='green')
    ax2.plot(linear_z, label='Linear Z', color='red')
    ax2.set_title('Linear Velocity')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 2D Plot: Angular Velocity (Bottom Right)
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(angular_x, label='Angular X', color='purple')
    ax3.plot(angular_y, label='Angular Y', color='orange')
    ax3.plot(angular_z, label='Angular Z', color='brown')
    ax3.set_title('Angular Velocity')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Velocity')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    
def visualize_data_with_3d_lidar_timeseries(dataset, num_samples=100):
    """
    Visualizes LiDAR data, position, linear velocity, and angular velocity,
    along with an animated subplot showing LiDAR points and position over time.
    Also includes a 3D LiDAR scan data time series where:
    - X-axis: Beam Index
    - Y-axis: Sample Index (Time Step)
    - Z-axis: Distance

    Parameters:
        dataset (LidarOdomDataset): The dataset to visualize.
        num_samples (int): Number of samples to display.
    """
    num_samples = min(num_samples, len(dataset.data))

    # Extract relevant data
    lidar_data = np.stack(dataset.data['ranges_list'].values[:num_samples])
    pos_x = dataset.data['pos_x'].values[:num_samples]
    pos_y = dataset.data['pos_y'].values[:num_samples]
    pos_z = dataset.data['pos_z'].values[:num_samples]

    linear_x = dataset.data['linear_x'].values[:num_samples]
    linear_y = dataset.data['linear_y'].values[:num_samples]
    linear_z = dataset.data['linear_z'].values[:num_samples]

    angular_x = dataset.data['angular_x'].values[:num_samples]
    angular_y = dataset.data['angular_y'].values[:num_samples]
    angular_z = dataset.data['angular_z'].values[:num_samples]

    #theta = np.linspace(np.pi, -np.pi, lidar_data.shape[1])  # Adjusting theta to ensure alignment
    theta = np.linspace(0, 2 * np.pi, lidar_data.shape[1])
    # Create figure with modified grid layout
    fig = plt.figure(figsize=(16, 12))
    grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2, 1], wspace=0.3, hspace=0.4)

    # 3D Plot: LiDAR Points & Position (Top Left)
    ax1 = fig.add_subplot(grid[0, 0], projection='3d')
    for i in range(num_samples):
        x = lidar_data[i] * np.cos(theta)
        y = lidar_data[i] * np.sin(theta)
        z = np.full_like(x, pos_z[i])
        ax1.plot(x, y, z, alpha=0.5)
    ax1.scatter(pos_x, pos_y, pos_z, c=pos_z, cmap='coolwarm', edgecolor='k', s=50)
    ax1.set_title('LiDAR Points & Position')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Animated 3D Plot: LiDAR points over time (Top Right)
    ax_anim = fig.add_subplot(grid[0, 1], projection='3d')
    ax_anim.set_xlim([-8, 8])
    ax_anim.set_ylim([-8, 8])
    ax_anim.set_zlim([-1, 1])
    ax_anim.set_title('Animated LiDAR Points & Position')

    lidar_plot, = ax_anim.plot([], [], [], 'b.', markersize=5)
    position_plot, = ax_anim.plot([], [], [], 'ro', markersize=5)
    trace_line, = ax_anim.plot([], [], [], 'r-', linewidth=2, alpha=0.7)  # Tracing line

    def update(frame):
        x = lidar_data[frame] * -np.cos(theta)
        y = lidar_data[frame] * np.sin(theta)
        z = np.full_like(x, pos_z[frame])

        lidar_plot.set_data(x, y)
        lidar_plot.set_3d_properties(z)

        position_plot.set_data([pos_x[frame]], [pos_y[frame]])
        position_plot.set_3d_properties([pos_z[frame]])

        # Update tracing line
        trace_line.set_data(pos_x[:frame+1], pos_y[:frame+1])
        trace_line.set_3d_properties(pos_z[:frame+1])

        return lidar_plot, position_plot, trace_line

    ani = animation.FuncAnimation(fig, update, frames=num_samples, interval=100, blit=True)

    # 2D Plot: Linear Velocity (Middle Left)
    ax2 = fig.add_subplot(grid[1, 0])
    ax2.plot(linear_x, label='Linear X', color='blue')
    ax2.plot(linear_y, label='Linear Y', color='green')
    ax2.plot(linear_z, label='Linear Z', color='red')
    ax2.set_title('Linear Velocity')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 2D Plot: Angular Velocity (Middle Right)
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(angular_x, label='Angular X', color='blue')
    ax3.plot(angular_y, label='Angular Y', color='green')
    ax3.plot(angular_z, label='Angular Z', color='red')
    ax3.set_title('Angular Velocity')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Velocity')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 3D Plot: LiDAR Scan Data Time Series (Bottom)
    ax4 = fig.add_subplot(grid[0, 2], projection='3d')
    beam_indices = np.arange(lidar_data.shape[1])
    sample_indices = np.arange(num_samples)

    for i in range(num_samples):
        ax4.plot(beam_indices, lidar_data[i], zs=i, zdir='y', alpha=0.5)

    ax4.set_title('1D LiDAR Scan Data Time Series')
    ax4.set_xlabel('Beam Index')
    ax4.set_ylabel('Sample Index')
    ax4.set_zlabel('Distance')
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"
    
    dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)
    #visualize_data(dataset, num_samples=100)
    visualize_data_with_3d_lidar_timeseries(dataset, num_samples=100)


