import os
import sys
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState

# Add parent directory to sys.path if necessary.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import LidarOdomDataset
from splitting import split_dataset

########################################
# Logging Setup
########################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(console)

########################################
# Accuracy Calculator Function
########################################
def calc_accuracy_percentage_xy(gt_array: np.ndarray,
                                pred_array: np.ndarray,
                                x_thresh: float = 0.1,
                                y_thresh: float = 0.1):
    """
    Compares ground truth vs. predicted 2D positions.
    Expects arrays of shape (N,2) for x and y.
    Returns the accuracy percentage (in %), along with thresholds.
    """
    N = gt_array.shape[0]
    if N == 0:
        return 0.0, x_thresh, y_thresh

    correct = 0
    for i in range(N):
        x_gt, y_gt = gt_array[i]
        x_pd, y_pd = pred_array[i]
        err_x = abs(x_pd - x_gt)
        err_y = abs(y_pd - y_gt)
        if err_x <= x_thresh and err_y <= y_thresh:
            correct += 1
    accuracy = 100.0 * correct / N
    return accuracy, x_thresh, y_thresh

########################################
# Candidate Mamba-based Model without explicit PositionalEncoding
########################################
class MambaNet(nn.Module):
    """
    MambaNet for LiDAR localization.
    Expects input of shape (B, input_size, in_features), where:
      - input_size is the number of LiDAR beams (e.g., 360),
      - in_features is the number of features per beam (default 2).
    This version does not use a fixed PositionalEncoding module.
    Instead, after embedding, a lightweight 1D convolution with circular padding
    mixes local positional information.
    
    The resulting sequence is then processed through a stack of Mamba blocks,
    followed by global average pooling and a final FC layer to produce (pos_x, pos_y, orientation_z).
    
    Note: The Mamba block is imported from mamba_ssm.
    """
    def __init__(self, input_size, output_size, in_features=2, d_model=64,
                 num_blocks=3, dropout=0.1, d_state=16, d_conv=2, expand=2):
        """
        Args:
          input_size: number of LiDAR beams (e.g., 360)
          output_size: number of regression outputs (3)
          in_features: features per beam (default 2)
          d_model: embedding dimension
          num_blocks: number of Mamba blocks
          dropout: dropout rate after each block
          d_state, d_conv, expand: hyperparameters for each Mamba block.
                         (Note: d_conv must be between 2 and 4)
        """
        super(MambaNet, self).__init__()
        self.input_size = input_size
        self.in_features = in_features
        self.d_model = d_model
        
        # Embedding layer: maps in_features to d_model.
        self.embedding = nn.Linear(in_features, d_model)
        # Instead of fixed PositionalEncoding, use a 1D convolution to mix positional info.
        self.pos_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                  kernel_size=3, stride=1, padding=1, padding_mode="circular")
        
        # Stack Mamba blocks.
        from mamba_ssm import Mamba  # Ensure mamba_ssm is built and installed.
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                nn.Dropout(dropout)
            ) for _ in range(num_blocks)
        ])
        
        # Global pooling and regression head.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # If input is 2D (B, input_size), assume a single scan and expand to (B, input_size, in_features).
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            if self.in_features == 2:
                x = x.repeat(1, 1, 2)
        # x: (B, input_size, in_features)
        # 1. Embedding.
        x_embed = self.embedding(x)  # (B, input_size, d_model)
        # Transpose for convolution: (B, d_model, input_size)
        x_embed = x_embed.transpose(1, 2)
        # 2. Positional mixing via 1D CNN.
        x_embed = self.pos_conv(x_embed)  # (B, d_model, input_size)
        # Transpose back: (B, input_size, d_model)
        x_embed = x_embed.transpose(1, 2)
        
        # 3. Process through each Mamba block with residual connection.
        for block in self.blocks:
            x_embed = x_embed + block(x_embed)
        
        # 4. Global pooling.
        x_embed = x_embed.transpose(1, 2)  # (B, d_model, input_size)
        pooled = self.pool(x_embed).squeeze(-1)  # (B, d_model)
        # 5. Output layer.
        return self.fc(pooled)  # (B, output_size)

########################################
# 1) DataLoaders
########################################
def create_dataset_and_loaders():
    dataset = LidarOdomDataset("dataset/odom_data.csv", "dataset/scan_data.csv")
    train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.2, 0.1)
    input_size = len(dataset[0][0])   # e.g., 360 beams
    output_size = len(dataset[0][1])   # e.g., 3 outputs
    return train_ds, val_ds, test_ds, input_size, output_size

########################################
# 2) Partial Training with Pruning and Accuracy Calculation
########################################
def train_and_eval(model, train_loader, val_loader, device, trial, max_epochs=10):
    # Use L1Loss as criterion.
    criterion = nn.L1Loss()
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch in range(max_epochs):
        train_loss = 0.0
        for lidar_batch, odom_batch in train_loader:
            lidar_batch = lidar_batch.to(device).float()
            odom_batch = odom_batch.to(device).float()
            optimizer.zero_grad()
            preds = model(lidar_batch)
            loss = criterion(preds, odom_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation loop.
        val_loss = 0.0
        all_preds = []
        all_gts = []
        model.eval()
        with torch.no_grad():
            for lidar_batch, odom_batch in val_loader:
                lidar_batch = lidar_batch.to(device).float()
                odom_batch = odom_batch.to(device).float()
                preds = model(lidar_batch)
                loss = criterion(preds, odom_batch)
                val_loss += loss.item()
                all_preds.append(preds.cpu())
                all_gts.append(odom_batch.cpu())
        val_loss /= len(val_loader)
        model.train()
        
        # Concatenate predictions and ground truth.
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_gts = torch.cat(all_gts, dim=0).numpy()
        # Calculate accuracy based on first two coordinates (x, y).
        accuracy, _, _ = calc_accuracy_percentage_xy(all_gts[:, :2], all_preds[:, :2])
        
        trial.report(val_loss, epoch)
        trial.set_user_attr("accuracy", accuracy)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        #logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.10f}, Accuracy: {accuracy:.0f}%")
    
    return val_loss

########################################
# 3) Objective Function for Optuna
########################################
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Architecture hyperparameters.
    d_model = trial.suggest_int("d_model", 32, 128, step=16)
    num_blocks = trial.suggest_int("num_blocks", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    d_state = trial.suggest_int("d_state", 8, 32, step=4)
    # d_conv must be between 2 and 4.
    d_conv = trial.suggest_int("d_conv", 2, 4, step=1)
    expand = trial.suggest_int("expand", 2, 4)
    
    # Build the model.
    model = MambaNet(input_size=input_size, output_size=output_size,
                     in_features=2, d_model=d_model, num_blocks=num_blocks,
                     dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand).to(device)
    
    max_epochs = 20
    val_loss = train_and_eval(model, train_loader, val_loader, device, trial, max_epochs)
    return val_loss

########################################
# 4) Main: Setup Study and Run Optimization
########################################
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_ds, val_ds, test_ds, input_size, output_size = create_dataset_and_loaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    globals()['train_ds'] = train_ds
    globals()['val_ds']   = val_ds
    globals()['test_ds']  = test_ds
    globals()['input_size'] = input_size
    globals()['output_size'] = output_size
    globals()['device']      = device
    
    storage_url = "sqlite:///mamba_search2.db"
    study_name = "mamba_search2"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    N_TRIALS = 5
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    
    # Print a summary for each complete trial.
    logger.info("\nTrial summaries:")
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            acc = trial.user_attrs.get("accuracy", 0)
            logger.info(f"Trial {trial.number} finished with value: {trial.value:.10f}, "
                        f"accuracy: {acc:.0f}% and parameters: {trial.params}")
    
    best_trial = study.best_trial
    best_acc = best_trial.user_attrs.get("accuracy", 0)
    logger.info(f"\nBest is trial {best_trial.number} with value: {best_trial.value:.10f} and accuracy: {best_acc:.0f}%")
    
    logger.info("Run: optuna-dashboard sqlite:///mamba_search2.db")
