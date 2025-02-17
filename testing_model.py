# visualize_test.py
import csv
import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from architectures.architectures import (
    SimpleMLP, MLP_Optuna,
    Conv1DNet, Conv1DNet_Optuna,
    Conv1DLSTMNet, CNNLSTMNet_Optuna,
    ConvTransformerNet, CNNTransformerNet_Optuna
)
from utils.utils import calc_accuracy_percentage_xy
#######################################
# 1) CSV Reading
#######################################
def read_odom_csv(odom_csv_path):
    """
    Reads odom_data.csv with columns:
      sec,nanosec,pos_x,pos_y,pos_z,orientation_x,orientation_y,orientation_z,...
    We store { (sec,nanosec) : { "x":..., "y":..., "yaw":... } }.

    If orientation_z is not truly yaw, convert from quaternion or store raw if needed.
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
    returns list of dict: { 'key':(sec,nanosec), 'ranges': np.array([...]) }
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

#######################################
# 2) Model Loading + Inference
#######################################
def load_model(model_path, model_choice, input_size=360, output_size=3, device='cpu'):
    if model_choice == 'MLP':
        model = MLP_Optuna(input_size=input_size, output_size=output_size)
    elif model_choice == 'Conv1DNet':
        model = Conv1DNet_Optuna(input_size=input_size, output_size=output_size)
    elif model_choice == 'CNNLSTMNet_Optuna':
        model = CNNLSTMNet_Optuna(input_size=input_size, output_size=output_size)
    elif model_choice == 'CNNTransformerNet':
        model = CNNTransformerNet_Optuna(output_size=output_size)
    else:
        model = ConvTransformerNet(input_size=input_size, output_size=output_size)

    # Suppress the future pickle warning by explicitly specifying `weights_only=True` if you only saved weights
    # For example:
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
   #model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    return model

def run_inference(model, scan_array, device='cpu'):
    """
    For a single LiDAR scan: returns predicted (x, y, yaw)
    """
    inp = torch.from_numpy(scan_array).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = model(inp)  # shape [1,3]
    pred = out.cpu().numpy()[0]  # (3,)
    return pred[0], pred[1], pred[2]

#######################################
# 4) Single-plot Animation
#######################################
def animate_singleplot(
    odom_dict,
    scan_list,
    model,
    device='cpu',
    max_frames=300,
    save_path=None
):
    # Merge data
    data = []
    for item in scan_list:
        key = item['key']
        if key in odom_dict:
            x_gt = odom_dict[key]['x']
            y_gt = odom_dict[key]['y']
            yaw_gt = odom_dict[key]['yaw']
            data.append({
                "key": key,
                "scan": item['ranges'],
                "x_gt": x_gt,
                "y_gt": y_gt,
                "yaw_gt": yaw_gt
            })
    data.sort(key=lambda d: d["key"])
    if max_frames is not None:
        data = data[:max_frames]

    # Precompute
    results = []
    gt_list = []
    pred_list = []
    for d in data:
        x_pred, y_pred, yaw_pred = run_inference(model, d["scan"], device=device)
        results.append({
            "key": d["key"],
            "x_gt": d["x_gt"],
            "y_gt": d["y_gt"],
            "yaw_gt": d["yaw_gt"],
            "x_pred": x_pred,
            "y_pred": y_pred,
            "yaw_pred": yaw_pred,
            "scan": d["scan"]
        })
        # For accuracy arrays
        gt_list.append([d["x_gt"], d["y_gt"], d["yaw_gt"] ])
        pred_list.append([x_pred, y_pred, yaw_pred])

    # Convert to NumPy for accuracy
    gt_values = np.array(gt_list)    # shape (N,3)
    pred_values = np.array(pred_list)
    acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(gt_values, pred_values, 0.1, 0.1)
    print(f"Overall Accuracy: {acc_perc:.2f}%")

    # keep a running path for GT and Pred
    gt_x_vals, gt_y_vals = [], []
    pd_x_vals, pd_y_vals = [], []

    fig, ax = plt.subplots(figsize=(7,7))

    gt_line, = ax.plot([], [], 'b-', label='Ground Truth')
    pd_line, = ax.plot([], [], 'r-', label='Predicted')
    scan_scatter = ax.scatter([], [], c='r', s=1, alpha=0.7)

    # Add text to show overall accuracy
    # We'll place it in top-left corner
    acc_text = ax.text(
        0.02, 0.98, 
        f"Accuracy: {acc_perc:.2f}%\n"
        f"Max frames: {max_frames:.2f}\n"
        f"X threshold: {x_thresh:.2f}\n"
        f"Y threshold: {y_thresh:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )

    ax.legend(loc='lower right', framealpha=0.7)
    ax.set_title("Comparison GT vs Pred with LiDAR points")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)

    def update_limits():
        # Expand axes to fit new points
        all_x = gt_x_vals + pd_x_vals
        all_y = gt_y_vals + pd_y_vals
        if len(all_x) < 2:
            return
        margin = 3.0
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

    def init():
        gt_line.set_data([], [])
        pd_line.set_data([], [])
        scan_scatter.set_offsets(np.empty((0, 2)))
        return gt_line, pd_line, scan_scatter

    def animate(i):
        r = results[i]

        gt_x_vals.append(r["x_gt"])
        gt_y_vals.append(r["y_gt"])
        pd_x_vals.append(r["x_pred"])
        pd_y_vals.append(r["y_pred"])

        gt_line.set_data(gt_x_vals, gt_y_vals)
        pd_line.set_data(pd_x_vals, pd_y_vals)

        #Compute LiDAR points in global frame (simple assumption: angles in [-pi, pi])
        # ranges = r["scan"]
        # N = len(ranges)
        # # Create angles covering the full circle; adjust as needed
        # angles = np.linspace(math.pi, -math.pi, N, endpoint=False)
        # #angles = np.rad2deg(angles)
        
        # #global_angles = angles - r["yaw_gt"]
        # x0, y0 = r["x_gt"], r["y_gt"]
        # x_loc = ranges * np.cos(angles)
        # y_loc = ranges * np.sin(angles)

        # x_glob = x0 + x_loc
        # y_glob = y0 + y_loc
        # scan_pts = np.column_stack((x_glob, y_glob))
        #Use ground truth robot pose
        # pose = np.array([r["x_gt"], r["y_gt"], r["yaw_gt"]])
        # pose2 = np.array([r["x_pred"], r["y_pred"], r["yaw_pred"]])
        # # Convert polar coordinates to local Cartesian coordinates.
        # local_scan = pol2cart(angles, ranges)
        
        # # Transform local scan points to global frame.
        # global_scan = localToGlobal(pose, local_scan)
        #scan_scatter.set_offsets(scan_pts)

        update_limits()
        return gt_line, pd_line, scan_scatter

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(results),
        interval=100,
        init_func=init,
        blit=False
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=10)
        print("Animation saved!")

    plt.show()

#######################################
# Main
#######################################
def main():
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"
    model_path    = "models/CNNLSTMNet_Optuna_lr0.0006829381720401536_bs16_20250127_144344.pth"
    model_choice  = "CNNLSTMNet_Optuna"
    device        = "cuda"

    odom_dict = read_odom_csv(odom_csv_path)
    scan_list = read_scan_csv(scan_csv_path)

    model = load_model(model_path, model_choice, input_size=360, output_size=3, device=device)

    animate_singleplot(
        odom_dict=odom_dict,
        scan_list=scan_list,
        model=model,
        device=device,
        max_frames=300,
        #save_path="images/fig.mp4"
    )

if __name__ == "__main__":
    main()
