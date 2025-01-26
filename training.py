import time
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import LidarOdomDataset
from architectures_og import (
    SimpleMLP, DeeperMLP, Conv1DNet, 
    Conv1DLSTMNet, ConvTransformerNet, TransformerRegressor
)
from splitting import split_dataset
from utils.utils import visualize_test_loader_static, calc_accuracy_percentage_xy
from torch.optim.lr_scheduler import OneCycleLR

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def train_model(
    odom_csv='odom_data.csv',
    scan_csv='scan_data.csv',
    model_choice='simple_mlp',
    batch_size=32,
    lr=3e-4,
    epochs=10,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    random_seed=42,
    log_dir=None,
    do_visualize=False
):
    """Train model with accuracy tracking."""
    # TensorBoard setup
    if log_dir is None:
        log_dir = f"tensorboard_logs/{model_choice}_lr{lr}_bs{batch_size}"
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard writer created at {log_dir}")

    # Dataset setup
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder '{dataset_folder}' not found")

    odom_csv_path = os.path.join(dataset_folder, odom_csv)
    scan_csv_path = os.path.join(dataset_folder, scan_csv)

    if not all(os.path.exists(p) for p in [odom_csv_path, scan_csv_path]):
        raise FileNotFoundError("Dataset files not found")

    # Load and split dataset
    full_dataset = LidarOdomDataset(odom_csv_path, scan_csv_path)
    sample_lidar, _ = full_dataset[0]
    input_size = len(sample_lidar)
    logger.info(f"Input size: {input_size}")

    train_subset, val_subset, test_subset = split_dataset(
        full_dataset, train_ratio, val_ratio, test_ratio, random_seed
    )
    logger.info(f"Dataset split - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_classes = {
        'SimpleMLP': lambda: SimpleMLP(input_size=input_size, hidden1=128, hidden2=64, output_size=3),
        'DeeperMLP': lambda: DeeperMLP(input_size=input_size, hidden1=256, hidden2=128, hidden3=64, output_size=3),
        'Conv1DNet': lambda: Conv1DNet(input_size=input_size, output_size=3),
        'Conv1DLSTMNet': lambda: Conv1DLSTMNet(input_size=input_size, output_size=3),
        'ConvTransformerNet': lambda: ConvTransformerNet(input_size=input_size, output_size=3),
        'TransformerRegressor': lambda: TransformerRegressor(input_size, output_size=3)
    }

    model = model_classes[model_choice]()
    model.to(device)
    logger.info(f"Initialized {model_choice} on {device}")

    # Training setup
    criterion = nn.HuberLoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,              # Start higher - like beginning a journey at a good cruising speed
        weight_decay=0.1,    
        betas=(0.9, 0.95)
    )


    ##############################################
    # ReduceLROnPlateau
    ##############################################  
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,          # More gentle reduction - halving instead of reducing by 80%
    #     patience=7,          # Give more time to find improvements
    #     min_lr=1e-6,        # Keep this as is
    #     verbose=True,
    #     threshold=0.01,      # Consider an improvement significant if loss decreases by 1%
    #     cooldown=3          # Wait 3 epochs after each reduction before reducing again
    # )

    ##############################################
    # OneCycleLR
    #############################################
    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-4,  # Higher initial rate
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Warm-up for 20% of training
        div_factor=20,  # Initial lr = max_lr/20
        final_div_factor=1e4
    )

    # Training state tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_accuracy = 0.0
    best_model_state = None
    early_stop_patience = 60
    early_stop_counter = 0

    os.makedirs("models", exist_ok=True)

    # Training loop
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        gt_list = []
        pd_list = []

        with torch.no_grad():
            for lidar_batch, odom_batch in val_loader:
                lidar_batch = lidar_batch.to(device)
                odom_batch = odom_batch.to(device)
                preds = model(lidar_batch)
                val_loss = criterion(preds, odom_batch)
                total_val_loss += val_loss.item()

                gt_list.extend(odom_batch.cpu().numpy())
                pd_list.extend(preds.cpu().numpy())

        # Calculate metrics
        avg_val_loss = total_val_loss / len(val_loader)
        acc_perc, x_thresh, y_thresh = calc_accuracy_percentage_xy(
            np.array(gt_list), np.array(pd_list)
        )

        # Update tracking
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(acc_perc)

        # Model saving logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = f"models/{model_choice}_best_val_loss_epoch{epoch+1}_loss{avg_val_loss:.4f}.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model (loss) saved: {model_path}")

        if acc_perc > best_accuracy:
            best_accuracy = acc_perc
            model_path = f"models/{model_choice}_best_accuracy_epoch{epoch+1}_acc{acc_perc:.2f}.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model (accuracy) saved: {model_path}")

        # Early stopping check
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Update learning rate
        # scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, "
            f"Accuracy={acc_perc:.2f}%, "
            f"LR={current_lr:.6g}, "
            f"Time={time.time() - start_time:.2f}s"
        )

        # TensorBoard logging
        writer.add_scalar("Train Loss", avg_train_loss, epoch)
        writer.add_scalar("Val Loss", avg_val_loss, epoch)
        writer.add_scalar("Val Accuracy", acc_perc, epoch)
        writer.add_scalar("Learning Rate", current_lr, epoch)

    # Final evaluation
    model.eval()
    test_loss = 0.0
    test_gt = []
    test_pd = []

    with torch.no_grad():
        for lidar_batch, odom_batch in test_loader:
            lidar_batch = lidar_batch.to(device)
            odom_batch = odom_batch.to(device)
            preds = model(lidar_batch)
            loss = criterion(preds, odom_batch)
            test_loss += loss.item()
            test_gt.extend(odom_batch.cpu().numpy())
            test_pd.extend(preds.cpu().numpy())

    final_test_loss = test_loss / len(test_loader)
    final_accuracy, _, _ = calc_accuracy_percentage_xy(
        np.array(test_gt), np.array(test_pd)
    )

    logger.info(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_accuracy:.2f}%")
    writer.close()

    if do_visualize:
        logger.info("Generating visualization...")
        visualize_test_loader_static(model, test_loader, device=device, max_samples=len(test_subset))

    return final_test_loss, final_accuracy, best_accuracy

def main():
    final_loss, test_acc, best_acc = train_model(
        model_choice='ConvTransformerNet', # ConvTransformerNet
        batch_size=32,
        lr=5e-4,
        epochs=100,
        do_visualize=True
    )
    logger.info(
        f"Training complete - "
        f"Final Loss: {final_loss:.4f}, "
        f"Test Accuracy: {test_acc:.2f}%, "
        f"Best Val Accuracy: {best_acc:.2f}%"
    )

if __name__ == "__main__":
    main()