import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # <-- For TensorBoard
import os
from dataset import LidarOdomDataset
from architectures import (
    SimpleMLP,
    DeeperMLP,
    Conv1DNet,
    Conv1DLSTMNet,
    ConvTransformerNet
)
from splitting import split_dataset

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

# ---------------------------
# Check GPU
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

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
    log_dir=None # TensorBoard logdir
):
    """
    Train a LiDAR->Velocity model. Returns final test loss, epoch_times.
    Also logs data to TensorBoard (SummaryWriter) and Python logger.
    """
    # ---------------------------
    # TensorBoard Setup
    # ---------------------------
    if log_dir is None:  # Dynamically format the log_dir
        log_dir = f"tensorboard_logs/{model_choice}_lr{lr}_bs{batch_size}"
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard writer created at {log_dir}")
    # ---------------------------
    # Ensure dataset folder exists and retrieve files
    # ---------------------------
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"The dataset folder '{dataset_folder}' does not exist. Please create it and add the dataset files.")

    # Set paths to dataset files inside the dataset folder
    odom_csv_path = os.path.join(dataset_folder, odom_csv)
    scan_csv_path = os.path.join(dataset_folder, scan_csv)

    # Ensure the dataset files exist
    if not os.path.exists(odom_csv_path):
        raise FileNotFoundError(f"Odometry data file '{odom_csv_path}' not found in '{dataset_folder}'.")
    if not os.path.exists(scan_csv_path):
        raise FileNotFoundError(f"LaserScan data file '{scan_csv_path}' not found in '{dataset_folder}'.")

    # ---------------------------
    # 1. Load dataset
    # ---------------------------
    full_dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)

    # 2. Inspect sample for input_size
    sample_lidar, _ = full_dataset[0]  # shape: (N_lidar_beams,)
    input_size = len(sample_lidar)     # e.g., 360
    logger.info(f"Detected input_size={input_size} from sample data")

    # 3. Split dataset
    train_subset, val_subset, test_subset = split_dataset(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    logger.info(
        f"Dataset split into: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # 4. DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # 5. Model selection
    logger.info(f"Initializing model: {model_choice}")
    if model_choice == 'SimpleMLP':
        model = SimpleMLP(input_size=input_size, hidden1=128, hidden2=64, output_size=3)
    elif model_choice == 'DeeperMLP':
        model = DeeperMLP(input_size=input_size, hidden1=256, hidden2=128, hidden3=64, output_size=3)
    elif model_choice == 'Conv1DNet':
        model = Conv1DNet(input_size=input_size, output_size=3)
    elif model_choice == 'Conv1DLSTMNet':
        model = Conv1DLSTMNet(input_size=input_size, output_size=3)
    elif model_choice == 'ConvTransformerNet':
        model = ConvTransformerNet(input_size=input_size, output_size=3)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    model.to(device)
    logger.info(f"Model {model_choice} moved to {device}")

    # 6. Define loss & optimizer & scheduler
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8) # after each step_size = N update /// newLR = oldRL * gamma

    logger.info(f"Optimizer: AdamW, LR={lr}")
    logger.info("Using HuberLoss as training criterion")

    train_losses = []
    val_losses = []
    epoch_times = []

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(epochs):
        start_time = time.time()

        # Train phase
        model.train()
        total_train_loss = 0.0
        for lidar_batch, odom_batch in train_loader:
            lidar_batch = lidar_batch.to(device)
            odom_batch = odom_batch.to(device)

            optimizer.zero_grad()
            preds = model(lidar_batch)
            loss = criterion(preds, odom_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for lidar_batch, odom_batch in val_loader:
                lidar_batch = lidar_batch.to(device)
                odom_batch = odom_batch.to(device)
                preds = model(lidar_batch)
                val_loss = criterion(preds, odom_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step()

        # Time measurement
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # TensorBoard logging
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalar("Train Loss", avg_train_loss, epoch)
        writer.add_scalar("Val Loss", avg_val_loss, epoch)
        writer.add_scalar("Epoch Time", epoch_time, epoch)

        # Logging
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"TrainLoss={avg_train_loss:.4f}, "
            f"ValLoss={avg_val_loss:.4f}, "
            f"LR={current_lr:.6g}, "
            f"Time={epoch_time:.2f}s"
        )

    # ---------------------------
    # Final test evaluation
    # ---------------------------
    total_test_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for lidar_batch, odom_batch in test_loader:
            lidar_batch = lidar_batch.to(device)
            odom_batch = odom_batch.to(device)
            preds = model(lidar_batch)
            loss_test = criterion(preds, odom_batch)

            batch_sz = lidar_batch.size(0)
            total_test_loss += loss_test.item() * batch_sz
            total_samples += batch_sz

    final_test_loss = total_test_loss / total_samples if total_samples > 0 else 0.0
    logger.info(f"Final Test Loss: {final_test_loss:.4f}")

    # ---------------------------
    # Save model with descriptive name
    # ---------------------------
    os.makedirs("models", exist_ok=True)

    # Get current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Construct model file name
    model_filename = f"models/{model_choice}_lr{lr}_bs{batch_size}_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    logger.info(f"Model training complete. Saved to '{model_filename}'")


    # Close TensorBoard writer
    writer.close()

    # Return final test loss + epoch times
    return final_test_loss, epoch_times

def main():
    final_loss, epoch_times = train_model(
        odom_csv='odom_data.csv',
        scan_csv='scan_data.csv',
        model_choice='Conv1DNet',
        batch_size=64,
        lr=1e-4,
        epochs=200,
    )
    logger.info(f"Training script done. Final Loss: {final_loss:.4f}")
    #logger.info(f"Epoch times: {epoch_times}")

if __name__ == "__main__":
    main()
