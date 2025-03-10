# training.py
import time
import logging
import torch
import torch.nn as nn
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # <-- For TensorBoard
import os
from dataset import LidarOdomDataset
from diffusion import DiffusionRegressor, forward_diffusion
from splitting import split_dataset
from viz_diffusion import visualize_test_loader_static 
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
    log_dir=None, # TensorBoard logdir
    do_visualize=False
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
    if model_choice == 'DiffusionRegressor':
        model = DiffusionRegressor()
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
    T = 50  # or 100
    betas = torch.linspace(1e-4, 0.02, T).to(device)
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
            t = torch.randint(0, T, (odom_batch.size(0),), device=device)
            pose_t, eps = forward_diffusion(odom_batch, t, betas)
            pred_noise  = model(pose_t, lidar_batch, t)
            loss = criterion(pred_noise, eps)
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
                t = torch.randint(0, T, (odom_batch.size(0),), device=device)
                pose_t, eps = forward_diffusion(odom_batch, t, betas)
                pred_noise  = model(pose_t, lidar_batch, t)
                val_loss = criterion(pred_noise, eps)
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
            t = torch.randint(0, T, (odom_batch.size(0),), device=device)
            pose_t, eps = forward_diffusion(odom_batch, t, betas)
            pred_noise  = model(pose_t, lidar_batch, t)
            loss_test = criterion(pred_noise, eps)
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
    # If do_visualize => plot using the test_loader
    if do_visualize:
        logger.info("Starting Matplotlib animation on test_loader data...")
        visualize_test_loader_static(model, test_loader, device=device, max_samples=total_samples) 
        logger.info("Animation done.")


    # Return final test loss + epoch times
    return final_test_loss, epoch_times

def main():
    final_loss, epoch_times = train_model(
        odom_csv='odom_data.csv',
        scan_csv='scan_data.csv',
        model_choice='DiffusionRegressor', # SimpleMLP, DeeperMLP, Conv1DNet, Conv1DLSTMNet, ConvTransformerNet, TransformerRegressor
        batch_size=64,
        lr=1e-4,
        epochs=10,
        do_visualize=True
    )
    logger.info(f"Training script done. Final Loss: {final_loss:.4f}")
    #logger.info(f"Epoch times: {epoch_times}")

if __name__ == "__main__":
    main()
