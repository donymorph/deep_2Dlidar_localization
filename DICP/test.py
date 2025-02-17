import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import numpy as np

# Adjust the sys.path to include the parent directory if needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DICP.dicp_dataset import LidarOdomPairDataset
from DICP.dicp import LocalizationNet
from splitting import split_dataset_tyler
from utils.utils import setup_logger, get_device, calc_accuracy_percentage_xy, visualize_test_loader_static_dicp

logger = setup_logger()
device = get_device()

# ------------------------------------------------------------------
# Test 1: Dataset Loading and Splitting
# ------------------------------------------------------------------
def test_dataset_loading(odom_csv='odom_data.csv', scan_csv='scan_data.csv', gap=1, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=41):
    dataset_folder = "dataset"
    odom_csv_path = os.path.join(dataset_folder, odom_csv)
    scan_csv_path = os.path.join(dataset_folder, scan_csv)
    
    if not os.path.exists(odom_csv_path) or not os.path.exists(scan_csv_path):
        raise FileNotFoundError("One or both CSV files not found in the dataset folder.")
    
    full_dataset = LidarOdomPairDataset(odom_csv_path, scan_csv_path, gap=gap)
    total_samples = len(full_dataset)
    logger.info(f"Total paired samples in dataset: {total_samples}")
    
    # Use the provided splitting function.
    train_subset, val_subset, test_subset = split_dataset_tyler(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    logger.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}")
    
    # Print out information for the first sample.
    sample = full_dataset[0]
    lidar_input, gt_transformation, source_points, target_points = sample
    logger.info(f"Sample 0 - Lidar shape: {lidar_input.shape}, GT Transformation: {gt_transformation}, Source pts: {source_points.shape}, Target pts: {target_points.shape}")
    
    return train_subset, val_subset, test_subset

# ------------------------------------------------------------------
# Test 2: Run a Single Training Epoch and Log Losses
# ------------------------------------------------------------------
def test_training_loop(odom_csv='odom_data.csv', scan_csv='scan_data.csv', batch_size=16, lr=0.001, epochs=1):
    # Load and split data.
    train_subset, val_subset, test_subset = test_dataset_loading(odom_csv, scan_csv)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model.
    model = LocalizationNet().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    logger.info("Starting training loop test...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        for data in train_loader:
            # Unpack: lidar, odom (gt transformation), source, target.
            lidar_batch, odom_batch, source_points, target_points = data
            lidar_batch   = lidar_batch.to(device)
            odom_batch    = odom_batch.to(device)
            source_points = source_points.to(device)
            target_points = target_points.to(device)
            
            optimizer.zero_grad()
            preds = model(lidar_batch, source_points, target_points)
            # Reorder predicted output from [theta, tx, ty] to [tx, ty, theta]
            #preds = torch.stack([preds[:, 1], preds[:, 2], preds[:, 0]], dim=1)
            loss = criterion(preds, odom_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}]: Train Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
    
    # Evaluate on validation set.
    model.eval()
    val_loss = 0.0
    gt_list, pred_list = [], []
    with torch.no_grad():
        for data in val_loader:
            lidar_batch, odom_batch, source_points, target_points = data
            lidar_batch   = lidar_batch.to(device)
            odom_batch    = odom_batch.to(device)
            source_points = source_points.to(device)
            target_points = target_points.to(device)
            preds = model(lidar_batch, source_points, target_points)
            preds = torch.stack([preds[:, 1], preds[:, 2], preds[:, 0]], dim=1)
            loss = criterion(preds, odom_batch)
            val_loss += loss.item()
            gt_list.extend(odom_batch.cpu().numpy())
            pred_list.extend(preds.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    gt_array = np.array(gt_list)
    pred_array = np.array(pred_list)
    acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(gt_array, pred_array)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {acc_perc:.2f}%")
    
    return model, test_loader

# ------------------------------------------------------------------
# Test 3: Visualization of Test Samples
# ------------------------------------------------------------------
def test_visualization(model, test_loader, max_samples=10):
    logger.info(f"Visualizing {max_samples} test samples...")
    # This function uses your provided visualization function.
    visualize_test_loader_static_dicp(model, test_loader, device=device, max_samples=max_samples)

# ------------------------------------------------------------------
# Main Test Runner
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Run dataset loading test.
    logger.info("----- Running Dataset Loading Test -----")
    train_subset, val_subset, test_subset = test_dataset_loading()
    
    # Run training loop test (with 1 epoch for quick test).
    logger.info("----- Running Training Loop Test -----")
    model, test_loader = test_training_loop(epochs=1)
    
    # Visualize a portion of test samples.
    logger.info("----- Running Visualization Test -----")
    test_visualization(model, test_loader, max_samples=10)
    
    logger.info("All tests completed.")
