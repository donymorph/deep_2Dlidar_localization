import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import sys  
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from velocity.vel_dataset import LidarOdomDataset
from architectures.architectures import (
    SimpleMLP, MLP_Optuna,
    Conv1DNet, Conv1DNet_Optuna,
    Conv1DLSTMNet, CNNLSTMNet_Optuna,
    ConvTransformerNet, CNNTransformerNet_Optuna
)
from torch.optim.lr_scheduler import StepLR
from splitting import split_dataset_tyler
from utils.utils import visualize_test_loader_static, calc_accuracy_percentage_xy, setup_logger, get_device, setup_tensorboard
from velocity.helper import integrate_velocities, visualize_velocity_predictions
#from architectures.Mamba import MambaModel, Mamba2Model, MambaModel_simple
#from architectures.hybridCNN_LSTM import CNNLSTMNet_modified
from architectures.Transformers_hybrid import LiDARFormer
logger = setup_logger()
device = get_device()
# ---------------------------
# Load Dataset and Split
# ---------------------------
def load_and_split_data(odom_csv, scan_csv, train_ratio, val_ratio, test_ratio, random_seed):
    dataset_folder = "dataset"
    odom_csv_path = os.path.join(dataset_folder, odom_csv)
    scan_csv_path = os.path.join(dataset_folder, scan_csv)

    if not os.path.exists(odom_csv_path):
        raise FileNotFoundError(f"Odometry data file '{odom_csv_path}' not found in '{dataset_folder}'.")
    if not os.path.exists(scan_csv_path):
        raise FileNotFoundError(f"LaserScan data file '{scan_csv_path}' not found in '{dataset_folder}'.")

    full_dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)
    sample_lidar = full_dataset[0]  # Get a sample to determine input size
    input_size = 360 #len(sample_lidar)
    
    train_subset, val_subset, test_subset = split_dataset_tyler(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    return train_subset, val_subset, test_subset, input_size

# ---------------------------
# Initialize Model
# ---------------------------
def initialize_model(model_choice, input_size):
    model_dict = {
        'SimpleMLP': SimpleMLP,
        'MLP_Optuna': MLP_Optuna,
        'Conv1DNet_Optuna': Conv1DNet_Optuna,
        'CNNLSTMNet_Optuna': CNNLSTMNet_Optuna,
        'Conv1DLSTMNet': Conv1DLSTMNet,
        'CNNTransformerNet_Optuna': CNNTransformerNet_Optuna,
        #'CNNLSTMNet_modified': CNNLSTMNet_modified,
        'ConvTransformerNet': ConvTransformerNet,
        # 'MambaModel_simple': MambaModel_simple,
        # 'Mamba2Model': Mamba2Model,
        'LiDARFormer': LiDARFormer,
    }
    
    if model_choice not in model_dict:
        raise ValueError(f"Unknown model_choice: {model_choice}")
    
    model = model_dict[model_choice](input_size=input_size, output_size=3)
    return model

# ---------------------------
# Training Loop
# ---------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        lidar_batch, vel_batch, _, _ = batch  # unpack extra outputs
        lidar_batch, vel_batch = lidar_batch.to(device), vel_batch.to(device)
        optimizer.zero_grad()
        preds = model(lidar_batch)
        loss = criterion(preds, vel_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    gt_vel_list, pred_vel_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            lidar_batch, vel_batch, _, _ = batch
            lidar_batch, vel_batch = lidar_batch.to(device), vel_batch.to(device)
            preds = model(lidar_batch)
            loss = criterion(preds, vel_batch)
            total_val_loss += loss.item()
            gt_vel_list.extend(vel_batch.cpu().numpy())
            pred_vel_list.extend(preds.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)
    # You can compute a velocity-specific accuracy metric here if needed.
    return avg_val_loss  # , additional metrics if desired


def evaluate_test(model, test_loader, criterion, device):
    """
    Evaluates the model on test data by computing both the velocity loss and 
    the integration error after integrating the predicted velocities to positions.
    """
    model.eval()
    all_pred_vel = []
    all_gt_pos = []
    all_timestamps = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            lidar_batch, vel_batch, pos_batch, timestamp_batch = batch
            lidar_batch, vel_batch = lidar_batch.to(device), vel_batch.to(device)
            preds = model(lidar_batch)
            loss = criterion(preds, vel_batch)
            batch_size = lidar_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_pred_vel.extend(preds.cpu().numpy())
            all_gt_pos.extend(pos_batch.numpy())
            # Assume timestamp_batch is a tensor; convert to numpy array
            all_timestamps.extend(timestamp_batch.cpu().numpy())
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    # Sort results by timestamp (integration requires time order)
    all_pred_vel = np.array(all_pred_vel)
    all_gt_pos = np.array(all_gt_pos)
    all_timestamps = np.array(all_timestamps)
    sort_idx = np.argsort(all_timestamps)
    sorted_pred_vel = all_pred_vel[sort_idx]
    sorted_gt_pos = all_gt_pos[sort_idx]
    sorted_timestamps = all_timestamps[sort_idx]

    # Integrate predicted velocities to compute positions
    integrated_pred_pos = integrate_velocities(sorted_pred_vel, sorted_timestamps, sorted_gt_pos[0])
    # Compute an error metric (e.g., average L1 error) between integrated positions and ground truth positions
    integration_error = np.mean(np.abs(integrated_pred_pos - sorted_gt_pos))
    return avg_loss, integration_error


# ---------------------------
# Save the Model
# ---------------------------
def save_model(model, model_choice, lr, batch_size):
    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/{model_choice}_lr{lr}_bs{batch_size}_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    logger.info(f"Model training complete. Saved to '{model_filename}'")
    
# ---------------------------
# Train Model
# ---------------------------
def train_model(
    odom_csv='odom_data.csv',
    scan_csv='scan_data.csv',
    model_choice='simple_mlp',
    batch_size=32,
    lr=1e-3,
    epochs=10,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    random_seed=42,
    log_dir=None,
    do_visualize=False
):
    
    #writer = setup_tensorboard(log_dir, model_choice, lr, batch_size)

    # Load and split data
    train_subset, val_subset, test_subset, input_size = load_and_split_data(
        odom_csv, scan_csv, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = initialize_model(model_choice, input_size)
    model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
    
    # Training loop
    train_losses, val_losses, epoch_times = [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        avg_val_loss = validate_epoch(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        # Time measurement
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # TensorBoard logging
        # writer.add_scalar("Metrics/Accuracy", acc_perc, epoch)
        # writer.add_scalar("Metrics/Train Loss", avg_train_loss, epoch)
        # writer.add_scalar("Metrics/Val Loss", avg_val_loss, epoch)
        # writer.add_scalar("Time/Epoch Time", epoch_time, epoch)

        # Logging
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"TrainLoss={avg_train_loss:.4f}, "
            f"ValLoss={avg_val_loss:.4f}, "
            #f"Accuracy={acc_perc:.2f}%, "
            f"Time={epoch_time:.2f}s"
        )

    # Final test evaluation
    final_test_loss, integration_error = evaluate_test(model, test_loader, criterion, device)
    logger.info(f"Final Test Loss: {final_test_loss:.4f}")
    logger.info(f"Integration Error: {integration_error:.4f}")

    # Save the model
    #save_model(model, model_choice, lr, batch_size)

    # Visualize if required
    if do_visualize:
        visualize_velocity_predictions(model, test_loader, device=device, max_samples=300)#len(test_loader.dataset)

    # Close TensorBoard writer
    #writer.close()

    return final_test_loss, epoch_times

# ---------------------------
# Main Function
# ---------------------------
def main():
    final_loss, epoch_times = train_model(
        odom_csv='odom_data.csv',
        scan_csv='scan_data.csv',
        model_choice='LiDARFormer',  # Select model
        batch_size=16,
        lr=0.001,
        epochs=10,
        do_visualize=True
    )
    logger.info(f"Training script done. Final Loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()
