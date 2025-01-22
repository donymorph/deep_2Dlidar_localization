import csv
import numpy as np
from pysr import PySRRegressor
import logging
# ---------------------------
# Setup Python logging
# ---------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# You can configure handlers (console, file) as needed:
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Optional: add format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Step 1: Read Odometry Data
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
            #yaw_gt = float(row['orientation_z'])  # caution if quaternion

            odom_dict[key] = {
                "x": x_gt,
                "y": y_gt,
                #"yaw": yaw_gt
            }
    return odom_dict

# Step 2: Read Scan Data
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

# Step 3: Merge Data
def merge_data(odom_dict, scan_list):
    """
    Merge odometry and scan data based on the matching timestamp keys.
    Returns a dataset for regression.
    """
    X = []  # Input (scan ranges)
    Y = []  # Output (x, y, yaw)

    for scan in scan_list:
        key = scan['key']
        if key in odom_dict:
            odom = odom_dict[key]
            X.append(scan['ranges'])
            Y.append([odom['x'], odom['y'] ])#odom['yaw']

    return np.array(X), np.array(Y)

# Step 4: PySR Regression
def run_pysr(X, Y):
    equations = []
    for i, target in enumerate(["x", "y"]):
        logger.info(f"Fitting equation for {target}...")
        model = PySRRegressor(
            niterations=10,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["cos", "sin", "exp", "log", "sqrt"],
            elementwise_loss="loss(x, y) = abs(x - y) <= 1.0 ? 0.5 * (x - y)^2 : 1.0 * (abs(x - y) - 0.5 * 1.0)",  # Define the loss as MSE
            maxsize=10,
            verbosity=1,
            batching=True,
            batch_size=100,
        )
        model.fit(X, Y[:, i])
        logger.info(f"Best equation for {target}: {model.get_best()}")

        # Save the equations
        equations.append(model.equations_)
    return equations

# Step 5: Main Function
def main():
    # File paths (update these to your dataset paths)
    odom_csv_path = "dataset/odom_data.csv"
    scan_csv_path = "dataset/scan_data.csv"

    # Read data
    odom_dict = read_odom_csv(odom_csv_path)
    scan_list = read_scan_csv(scan_csv_path)

    # Merge data
    X, Y = merge_data(odom_dict, scan_list)

    # Run symbolic regression
    equations = run_pysr(X, Y)

    # Save and print the results
    for idx, model in enumerate(equations):
        target_name = ['x', 'y'][idx]  # Map indices to target names
        print(f"Best equation for {target_name}: {model}")

if __name__ == "__main__":
    main()
