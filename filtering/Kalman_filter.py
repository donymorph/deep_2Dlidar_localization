import argparse
import torch
from torchinfo import summary
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from architectures import (
    SimpleMLP, MLP_Optuna,
    Conv1DNet, Conv1DNet_Optuna,
    Conv1DLSTMNet, CNNLSTMNet_Optuna,
    ConvTransformerNet, CNNTransformerNet_Optuna
)
from dataset import LidarOdomDataset, LidarOdomDataset_Tyler
from torch.utils.data import DataLoader, TensorDataset
from filterpy.kalman import KalmanFilter
import numpy as np
from utils import utils

def get_model(model_choice:str, input_size:int):
    
    if model_choice == 'MLP_Optuna':
        model = MLP_Optuna(input_size=input_size, output_size=3)
    elif model_choice == 'Conv1DNet_Optuna':
        model = Conv1DNet_Optuna(input_size=input_size, output_size=3)
    elif model_choice == 'CNNLSTMNet_Optuna':
        model = CNNLSTMNet_Optuna(input_size=input_size, output_size=3)
    elif model_choice == 'CNNTransformerNet_Optuna':
        model = CNNTransformerNet_Optuna(output_size=3)
    elif model_choice == 'ConvTransformerNet':
        model = ConvTransformerNet(input_size=input_size, output_size=3)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    return model
    

def generate_predictions(model:torch.nn, batch_size:int, out_file:str, dataset, device="cuda") -> pd.DataFrame:

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # List to store all predictions
    all_preds = []
    model.eval()  # Set the model to evaluation mode if you're using it for inference
    with torch.no_grad():
        for lidar_batch, _ in data_loader:
            preds = model(lidar_batch.to(device))
            all_preds.append(preds.cpu())

    # Concatenate all the batch predictions
    all_preds = torch.cat(all_preds, dim=0).numpy()

    # Save the predictions to a CSV
    df = pd.DataFrame(all_preds, columns=['pos_x', 'pos_y', 'orientation_z'])
    df.to_csv(out_file, index=False)

    print(f"Predictions saved to {out_file}")

    return all_preds


# Initialize the Kalman filter
def init_kalman_filter():
    kf = KalmanFilter(dim_x=3, dim_z=3)
    dt = 1.0
    kf.F = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    kf.H = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    kf.x = np.array([[0], [0], [0]])
    kf.P = np.eye(3)
    kf.Q = np.eye(3) * 0.01
    kf.R = np.eye(3) * 0.1
    return kf


# Update the Kalman filter using the whole dataset
def filter_dataset(kf, predictions_csv:str):
    # Read the CSV file
    df = pd.read_csv(predictions_csv)

    # Check if the CSV file has exactly 3 columns
    if len(df.columns) != 3:
        raise ValueError("The CSV file must have exactly 3 columns.")

    dataset = df.values

    filtered_states = []
    for prediction in dataset:
        # Reshape the prediction to a column vector
        prediction = prediction.reshape(-1, 1)
        kf.predict()
        kf.update(prediction)
        filtered_states.append(kf.x.flatten())
    
    output_file_path = predictions_csv[:-3] + "kalman.csv"
    filtered_df = pd.DataFrame(filtered_states, columns=['pos_x', 'pos_y', 'orientation_z'])
    filtered_df.to_csv(output_file_path, index=False)

    print(f"Kalman filter results saved in {output_file_path}")

    return filtered_df


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/pred_data.csv", help="path to AI predictions")
    parser.add_argument("--model_path", type=str, help="path to the pre-trained model",
                        default="models/CNNTransformerNet_Optuna_lr6.89e-05_bs16_20250127_131437.pth")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- LOAD THE MODEL ------------------------------
    model_params = opt.model_path.split('/')[1]
    model_choice, optim, lr, bs, _, _ = model_params.split('_')
    batch_size = int(bs[2:])

    model = get_model(model_choice=f"{model_choice}_{optim}",
                      input_size=(batch_size, 360)).to(device=device)
    model.load_state_dict(torch.load(opt.model_path))

    # summary(model=model, input_size=(batch_size, 360), device="cuda")

    # --------------- LOAD PREDICTIONS DATASET --------------------
    dataset = LidarOdomDataset(odom_csv_path="dataset/odom_data.csv", scan_csv_path="dataset/scan_data.csv")
    odom_gt = dataset.data[['pos_x', 'pos_y', 'orientation_z']].to_numpy()
    dataset.save_odom_data(out_csv_filepath="dataset/processed_odom.csv")
   
    output_file_path = f"dataset/{model_choice}_{optim}_{lr}_{bs}.csv"
    preds = generate_predictions(model=model, batch_size=batch_size, dataset=dataset,
                                device=device, out_file=output_file_path)
    # Apply Filter
    kf = init_kalman_filter()
    filtered = filter_dataset(kf=kf, predictions_csv=output_file_path)

    model_acc = utils.calc_accuracy_percentage_xy(gt_array=odom_gt, pred_array=preds)
    kalman_acc =  utils.calc_accuracy_percentage_xy(gt_array=odom_gt, pred_array=filtered.to_numpy())

    print(preds[0].dtype)
    print("A C C U R A C Y")
    print(f"{model_acc[0]}% ===> {model_params}")
    print(f"{kalman_acc[0]}% ===> Kalman filter")
    
if __name__=="__main__":
    
    main()