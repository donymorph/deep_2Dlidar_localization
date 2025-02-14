import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset import LidarOdomDataset, LidarOdomDataset_Tyler, LidarOdomDataset_withNoise
from architectures import (
    SimpleMLP, MLP_Optuna,
    Conv1DNet, Conv1DNet_Optuna,
    Conv1DLSTMNet, CNNLSTMNet_Optuna,
    ConvTransformerNet, CNNTransformerNet_Optuna
)
from torch.optim.lr_scheduler import StepLR
from splitting import split_dataset_tyler
from utils.utils import visualize_test_loader_static, calc_accuracy_percentage_xy, setup_logger, get_device, setup_tensorboard

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
    sample_lidar, _ = full_dataset[0]  # Get a sample to determine input size
    input_size = len(sample_lidar)
    
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
        'CNNTransformerNet_Optuna': CNNTransformerNet_Optuna,
        'ConvTransformerNet': ConvTransformerNet,
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
    for lidar_batch, odom_batch in train_loader:
        lidar_batch, odom_batch = lidar_batch.to(device), odom_batch.to(device)
        optimizer.zero_grad()
        preds = model(lidar_batch)
        loss = criterion(preds, odom_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    gt_list, pd_list = [], []
    with torch.no_grad():
        for lidar_batch, odom_batch in val_loader:
            lidar_batch, odom_batch = lidar_batch.to(device), odom_batch.to(device)
            preds = model(lidar_batch)
            val_loss = criterion(preds, odom_batch)
            total_val_loss += val_loss.item()
            gt_list.extend(odom_batch.cpu().numpy())
            pd_list.extend(preds.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)
    acc_perc, _, _ = calc_accuracy_percentage_xy(np.array(gt_list), np.array(pd_list))
    return avg_val_loss, acc_perc
# ---------------------------
# Evaluate Test Data
# ---------------------------
def evaluate_test(model, test_loader, criterion, device):
    total_test_loss = 0.0
    total_samples = 0
    model.eval()
    
    with torch.no_grad():
        for lidar_batch, odom_batch in test_loader:
            lidar_batch, odom_batch = lidar_batch.to(device), odom_batch.to(device)
            preds = model(lidar_batch)
            loss_test = criterion(preds, odom_batch)
            batch_sz = lidar_batch.size(0)
            total_test_loss += loss_test.item() * batch_sz
            total_samples += batch_sz
    return total_test_loss / total_samples if total_samples > 0 else 0.0

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
    random_seed=41,
    log_dir=None,
    do_visualize=False
):
    
    writer = setup_tensorboard(log_dir, model_choice, lr, batch_size)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
    
    # Training loop
    train_losses, val_losses, epoch_times = [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        avg_val_loss, acc_perc = validate_epoch(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        # Time measurement
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # TensorBoard logging
        writer.add_scalar("Metrics/Accuracy", acc_perc, epoch)
        writer.add_scalar("Metrics/Train Loss", avg_train_loss, epoch)
        writer.add_scalar("Metrics/Val Loss", avg_val_loss, epoch)
        writer.add_scalar("Time/Epoch Time", epoch_time, epoch)

        # Logging
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"TrainLoss={avg_train_loss:.4f}, "
            f"ValLoss={avg_val_loss:.4f}, "
            f"Accuracy={acc_perc:.2f}%, "
            f"Time={epoch_time:.2f}s"
        )

    # Final test evaluation
    final_test_loss = evaluate_test(model, test_loader, criterion, device)
    logger.info(f"Final Test Loss: {final_test_loss:.4f}")

    # Save the model
    save_model(model, model_choice, lr, batch_size)

    # Visualize if required
    if do_visualize:
        visualize_test_loader_static(model, test_loader, device=device, max_samples=len(test_loader.dataset))

    # Close TensorBoard writer
    writer.close()

    return final_test_loss, epoch_times

# ---------------------------
# Main Function
# ---------------------------
def main():
    final_loss, epoch_times = train_model(
        odom_csv='odom_data.csv',
        scan_csv='scan_data.csv',
        model_choice='CNNTransformerNet_Optuna',  # Select model
        batch_size=16,
        lr=6.89e-5,
        epochs=200,
        do_visualize=True
    )
    logger.info(f"Training script done. Final Loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()
