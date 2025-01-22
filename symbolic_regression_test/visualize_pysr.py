import csv
import numpy as np
import matplotlib.pyplot as plt

def read_odom_csv(odom_csv_path):
    """
    Reads odom_data.csv with columns:
      sec,nanosec,pos_x,pos_y,pos_z,orientation_x,orientation_y,orientation_z
    """
    odom_dict = {}
    with open(odom_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = int(row['sec'])
            nanosec = int(float(row['nanosec']))
            key = (sec, nanosec)

            x_gt = float(row['pos_x'])
            y_gt = float(row['pos_y'])
            yaw_gt = float(row['orientation_z'])  # caution if quaternion

            odom_dict[key] = {
                "x": x_gt,
                "y": y_gt,
                "yaw": yaw_gt
            }
    return odom_dict

def read_scan_csv(scan_csv_path):
    """
    Reads scan_data.csv with columns:
      sec,nanosec,ranges
    """
    scan_list = []
    with open(scan_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = int(row['sec'])
            nanosec = int(float(row['nanosec']))
            key = (sec, nanosec)

            ranges_str = row['ranges'].strip().split()
            ranges_f = np.array([float(r) for r in ranges_str], dtype=np.float32)

            scan_list.append({
                "key": key,
                "ranges": ranges_f
            })
    return scan_list


############################################
# These functions must match the discovered
# symbolic expressions you want to visualize
############################################

def compute_x(x225, x105):
    """
    Example from the discovered expression:
        x_pred = 1.06588299241755 * sqrt(0.88019913*x236 + x86)
    """
    return np.sqrt(x225 + x105)

def compute_y(x134, x222, x332):
    """
    Example from the discovered expression:
        y_pred = sqrt(x134 + x222 + x332)
    """
    return np.sqrt(x134 + x222 + x332)

def compute_yaw(x52):
    """
    Example from the discovered expression:
        yaw_pred = exp(x52 / -0.10993844)
    """
    return np.exp(x52 / -0.10993844)


def visualize_trajectory_and_yaw(odom_data, scan_data, max_samples=500):
    """
    1) Plot the 2D trajectory (x vs. y) comparing ground truth to prediction.
    2) Plot yaw vs. sample index for ground truth vs. prediction.
    """
    # We will create a list of (time, x_gt, y_gt, yaw_gt, x_pred, y_pred, yaw_pred)
    data_list = []
    needed_indexes = [236, 86, 134, 222, 332, 52]
    max_needed_index = max(needed_indexes)

    count = 0
    for scan in scan_data:
        if count >= max_samples:
            break

        key = scan['key']
        if key not in odom_data:
            continue

        ranges = scan['ranges']
        if len(ranges) <= max_needed_index:
            # Not enough data in 'ranges' for the indexes we need
            continue

        x_gt = odom_data[key]['x']
        y_gt = odom_data[key]['y']
        yaw_gt = odom_data[key]['yaw']

        x_pred = compute_x(ranges[236], ranges[86])
        y_pred = compute_y(ranges[134], ranges[222], ranges[332])
        yaw_pred = compute_yaw(ranges[52])

        # Convert (sec, nanosec) to a single time value (just for ordering & plotting)
        sec, nanosec = key
        time_s = sec + nanosec * 1e-9

        data_list.append((time_s, x_gt, y_gt, yaw_gt, x_pred, y_pred, yaw_pred))
        count += 1

    # Sort the data list by time, just in case your CSV entries aren’t sequential
    data_list.sort(key=lambda x: x[0])

    # Unpack columns after sorting
    times = [item[0] for item in data_list]
    x_gt_list = [item[1] for item in data_list]
    y_gt_list = [item[2] for item in data_list]
    yaw_gt_list = [item[3] for item in data_list]
    x_pred_list = [item[4] for item in data_list]
    y_pred_list = [item[5] for item in data_list]
    yaw_pred_list = [item[6] for item in data_list]

    # -- Plot 1: 2D Trajectory --
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))

    ax[0].plot(x_gt_list, y_gt_list, 'b-', label='Ground Truth Path')
    ax[0].plot(x_pred_list, y_pred_list, 'r--', label='Predicted Path')
    ax[0].set_title("Trajectory in XY-plane")
    ax[0].set_xlabel("X position")
    ax[0].set_ylabel("Y position")
    ax[0].legend()
    ax[0].grid(True)

    # -- Plot 2: Yaw vs. Time (or sample index) --
    # We can just use sample index if you prefer, or the actual time from the messages.
    ax[1].plot(times, yaw_gt_list, 'b-', label='Ground Truth Yaw')
    ax[1].plot(times, yaw_pred_list, 'r--', label='Predicted Yaw')
    ax[1].set_title("Yaw over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Yaw (radians?)")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File paths
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"

    # Load datasets
    odom_data = read_odom_csv(odom_csv_path)
    scan_data = read_scan_csv(scan_csv_path)

    # Visualize
    visualize_trajectory_and_yaw(odom_data, scan_data, max_samples=500)
