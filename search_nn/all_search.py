import os
import math
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import LidarOdomDataset
from splitting import split_dataset
from utils.loss_functions import PoseLoss
import numpy as np

########################################
# Logging setup
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
# 1) Model Classes
########################################

# --- MLP ---
class MLP(nn.Module):
    """
    Basic MLP with optional dropout and an option to output features only.
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation_fn, dropout=0.0, features_only=False):
        super().__init__()
        layers = []
        prev_size = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(activation_fn())
            if dropout > 1e-6:
                layers.append(nn.Dropout(dropout))
            prev_size = h_size

        self.feature_dim = prev_size
        self.features_only = features_only

        if not features_only:
            layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- RNN ---
class RNNNet(nn.Module):
    """
    RNN-based model (supports LSTM/GRU/RNN) with an option for features_only.
    Assumes input is a vector which is unsqueezed into (batch, seq_len, feature=1).
    """
    def __init__(self, input_size, output_size, rnn_hidden_size, rnn_num_layers, rnn_dropout, rnn_type="LSTM", features_only=False):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=1, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, dropout=rnn_dropout, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=1, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, dropout=rnn_dropout, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size=1, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, nonlinearity="tanh", dropout=rnn_dropout, batch_first=True)
        self.feature_dim = rnn_hidden_size
        self.features_only = features_only
        if not features_only:
            self.fc = nn.Linear(rnn_hidden_size, output_size)

    def forward(self, x):
        # x: (batch, input_size) -> treat as sequence of length=input_size, feature=1
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        last_output = out[:, -1, :]  # (batch, rnn_hidden_size)
        if self.features_only:
            return last_output
        else:
            return self.fc(last_output)

# --- CNN1D ---
class CNN1DNet(nn.Module):
    """
    1D CNN model. Assumes input shape (batch, input_size) and unsqueezes to (batch, 1, input_size).
    """
    def __init__(self, input_size, output_size, num_filters, kernel_size, num_conv_layers, dropout, features_only=False):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = num_filters
        self.conv = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = num_filters
        self.features_only = features_only
        if not features_only:
            self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x):
        # x: (batch, input_size) -> (batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.global_pool(x).squeeze(-1)
        if self.features_only:
            return x
        else:
            return self.fc(x)

# --- Transformer ---
class TransformerNet(nn.Module):
    """
    Transformer-based model. Treats the input vector as a sequence of length=input_size with feature=1.
    """
    def __init__(self, input_size, output_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, features_only=False):
        super().__init__()
        self.input_linear = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = d_model
        self.features_only = features_only
        if not features_only:
            self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, input_size) -> (batch, input_size, 1)
        x = x.unsqueeze(-1)
        x = self.input_linear(x)  # (batch, input_size, d_model)
        x = x.transpose(0, 1)      # (input_size, batch, d_model) for transformer
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)      # (batch, input_size, d_model)
        x = x.transpose(1, 2)      # (batch, d_model, input_size)
        x = self.global_pool(x).squeeze(-1)
        if self.features_only:
            return x
        else:
            return self.fc(x)

# --- Combined Network ---
class CombinedNet(nn.Module):
    """
    Combines two (or more) branch networks by concatenating their feature outputs,
    then fusing them via a small FC network.
    """
    def __init__(self, branch_models, output_size, fusion_hidden_size):
        super().__init__()
        self.branch_models = nn.ModuleList(branch_models)
        total_feature_dim = sum([bm.feature_dim for bm in branch_models])
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size, output_size)
        )

    def forward(self, x):
        features = []
        for bm in self.branch_models:
            features.append(bm(x))
        x_cat = torch.cat(features, dim=1)
        return self.fusion(x_cat)

########################################
# 2) DataLoaders
########################################
def create_dataset_and_loaders():
    dataset = LidarOdomDataset("dataset/odom_data.csv", "dataset/scan_data.csv")
    train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.2, 0.1)
    input_size  = len(dataset[0][0])
    output_size = len(dataset[0][1])
    return train_ds, val_ds, test_ds, input_size, output_size

########################################
# 3) Partial Training with Pruning
########################################
def train_and_eval_model(model, train_loader, val_loader, device, trial, max_epochs=10):
    """
    Train model for up to max_epochs. At each epoch:
      - perform forward/backward on training set
      - measure validation loss
      - report to trial and check for pruning.
    """
    criterion = PoseLoss()
    # Choose LR and optimizer hyperparameters.
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
        
    best_accuracy = 0.0
    model.train()
    for epoch in range(max_epochs):
        train_loss = 0.0
        for lidar_batch, odom_batch in train_loader:
            lidar_batch = lidar_batch.to(device).float()
            odom_batch  = odom_batch.to(device).float()

            optimizer.zero_grad()
            preds = model(lidar_batch)
            loss  = criterion(preds, odom_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        val_loss = 0.0
        all_preds = []
        all_gts = []
        model.eval()
        with torch.no_grad():
            for lidar_batch, odom_batch in val_loader:
                lidar_batch = lidar_batch.to(device).float()
                odom_batch  = odom_batch.to(device).float()
                preds  = model(lidar_batch)
                val_loss += criterion(preds, odom_batch).item()
                all_preds.append(preds.cpu())
                all_gts.append(odom_batch.cpu())
        val_loss /= len(val_loader)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_gts   = torch.cat(all_gts,   dim=0).numpy()
        accuracy, _, _ = calc_accuracy_percentage_xy(all_gts[:, :2], all_preds[:, :2])
        best_accuracy = max(best_accuracy, accuracy)
        #trial.report(val_loss, epoch)
        trial.report(accuracy, epoch)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("training Loss", train_loss)
        trial.set_user_attr("validation Loss", val_loss)
        if trial.should_prune():
            raise optuna.TrialPruned()
        model.train()
    return best_accuracy #val_loss

########################################
# 4) Objective Function with Architecture Search
########################################
def objective(trial):
    # Create dataloaders with suggested batch_size
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Choose model architecture type with more combinations.
    model_type = trial.suggest_categorical("model_type", 
                    ["mlp", "rnn", "cnn1d", "transformer", 
                     "mlp+rnn", "mlp+cnn1d", "mlp+transformer", 
                     "rnn+cnn1d", "rnn+transformer", "cnn1d+transformer",
                     "mlp+rnn+cnn1d", "mlp+rnn+transformer", 
                     "mlp+cnn1d+transformer", "rnn+cnn1d+transformer",
                     "mlp+rnn+cnn1d+transformer"])

    # Single branch architectures
    if "+" not in model_type:
        if model_type == "mlp":
            num_hidden_layers = trial.suggest_int("mlp_num_hidden_layers", 1, 5)
            hidden_sizes = [trial.suggest_int(f"mlp_hidden_layer_{i}_size", 32, 256, step=32)
                            for i in range(num_hidden_layers)]
            activation_name = trial.suggest_categorical("mlp_activation_fn", ["ReLU", "Tanh", "LeakyReLU"])
            activation_fn = getattr(nn, activation_name)
            dropout = trial.suggest_float("mlp_dropout", 0.0, 0.5, step=0.1)
            model = MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                activation_fn=activation_fn,
                dropout=dropout,
                features_only=False
            ).to(device)

        elif model_type == "rnn":
            rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 32, 256, step=32)
            rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 3)
            rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.5, step=0.1)
            rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU", "RNN"])
            model = RNNNet(
                input_size=input_size,
                output_size=output_size,
                rnn_hidden_size=rnn_hidden_size,
                rnn_num_layers=rnn_num_layers,
                rnn_dropout=rnn_dropout,
                rnn_type=rnn_type,
                features_only=False
            ).to(device)

        elif model_type == "cnn1d":
            num_filters = trial.suggest_int("cnn_num_filters", 16, 128, step=16)
            kernel_size = trial.suggest_int("cnn_kernel_size", 3, 7, step=2)
            num_conv_layers = trial.suggest_int("cnn_num_conv_layers", 1, 3)
            cnn_dropout = trial.suggest_float("cnn_dropout", 0.0, 0.5, step=0.1)
            model = CNN1DNet(
                input_size=input_size,
                output_size=output_size,
                num_filters=num_filters,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers,
                dropout=cnn_dropout,
                features_only=False
            ).to(device)

        elif model_type == "transformer":
            # Sample nhead first.
            nhead = trial.suggest_int("transformer_nhead", 1, 4)
            # Compute multiplier bounds to ensure d_model is between 32 and 128.
            lower_mult = (32 + nhead - 1) // nhead  # Ceil(32/nhead)
            upper_mult = 128 // nhead
            d_model_mult = trial.suggest_int("transformer_d_model_mult", lower_mult, upper_mult)
            d_model = d_model_mult * nhead
            num_encoder_layers = trial.suggest_int("transformer_num_layers", 1, 3)
            dim_feedforward = trial.suggest_int("transformer_dim_feedforward", 64, 256, step=64)
            transformer_dropout = trial.suggest_float("transformer_dropout", 0.0, 0.5, step=0.1)
            model = TransformerNet(
                input_size=input_size,
                output_size=output_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
                features_only=False
            ).to(device)

    # Combination models: Build each branch and fuse features.
    else:
        branches = model_type.split("+")
        branch_models = []
        for branch in branches:
            if branch == "mlp":
                num_hidden_layers = trial.suggest_int("mlp_branch_num_hidden_layers", 1, 3)
                hidden_sizes = [trial.suggest_int(f"mlp_hidden_layer_{i}_size", 32, 256, step=32)
                                for i in range(num_hidden_layers)]
                activation_name = trial.suggest_categorical("mlp_branch_activation_fn", ["ReLU", "Tanh", "LeakyReLU"])
                activation_fn = getattr(nn, activation_name)
                dropout = trial.suggest_float("mlp_branch_dropout", 0.0, 0.5, step=0.1)
                branch_model = MLP(
                    input_size=input_size,
                    output_size=output_size,  # Not used when features_only=True
                    hidden_sizes=hidden_sizes,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    features_only=True
                )
                branch_models.append(branch_model)

            elif branch == "rnn":
                rnn_hidden_size = trial.suggest_int("rnn_branch_hidden_size", 32, 128, step=32)
                rnn_num_layers = trial.suggest_int("rnn_branch_num_layers", 1, 3)
                rnn_dropout = trial.suggest_float("rnn_branch_dropout", 0.0, 0.5, step=0.1)
                rnn_type = trial.suggest_categorical("rnn_branch_type", ["LSTM", "GRU", "RNN"])
                branch_model = RNNNet(
                    input_size=input_size,
                    output_size=output_size,
                    rnn_hidden_size=rnn_hidden_size,
                    rnn_num_layers=rnn_num_layers,
                    rnn_dropout=rnn_dropout,
                    rnn_type=rnn_type,
                    features_only=True
                )
                branch_models.append(branch_model)

            elif branch == "cnn1d":
                num_filters = trial.suggest_int("cnn_branch_num_filters", 16, 128, step=16)
                kernel_size = trial.suggest_int("cnn_branch_kernel_size", 3, 7, step=2)
                num_conv_layers = trial.suggest_int("cnn_branch_num_conv_layers", 1, 3)
                cnn_dropout = trial.suggest_float("cnn_branch_dropout", 0.0, 0.5, step=0.1)
                branch_model = CNN1DNet(
                    input_size=input_size,
                    output_size=output_size,
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    num_conv_layers=num_conv_layers,
                    dropout=cnn_dropout,
                    features_only=True
                )
                branch_models.append(branch_model)

            elif branch == "transformer":
                nhead = trial.suggest_int("transformer_branch_nhead", 2, 4, step=2)
                lower_mult = (32 + nhead - 1) // nhead
                upper_mult = 128 // nhead
                d_model_mult = trial.suggest_int("transformer_branch_d_model_mult", lower_mult, upper_mult)
                d_model = d_model_mult * nhead
                num_encoder_layers = trial.suggest_int("transformer_branch_num_layers", 1, 3)
                dim_feedforward = trial.suggest_int("transformer_branch_dim_feedforward", 64, 256, step=64)
                transformer_dropout = trial.suggest_float("transformer_branch_dropout", 0.0, 0.5, step=0.1)
                branch_model = TransformerNet(
                    input_size=input_size,
                    output_size=output_size,
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=transformer_dropout,
                    features_only=True
                )
                branch_models.append(branch_model)

        fusion_hidden_size = trial.suggest_int("fusion_hidden_size", 32, 128, step=32)
        model = CombinedNet(branch_models, output_size, fusion_hidden_size).to(device)

    # Partial training with pruning
    max_epochs = 200
    best_accuracy = train_and_eval_model(model, train_loader, val_loader, device, trial, max_epochs)
    return best_accuracy #val_loss
########################################
# Main
########################################
if __name__ == "__main__":
    # Create dataset and dataloaders
    train_ds, val_ds, test_ds, input_size, output_size = create_dataset_and_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Expose global variables for the objective function
    globals()['train_ds'] = train_ds
    globals()['val_ds']   = val_ds
    globals()['test_ds']  = test_ds
    globals()['input_size'] = input_size
    globals()['output_size'] = output_size
    globals()['device']      = device

    # Create or load study from SQLite with pruner enabled
    storage_url = "sqlite:///all_search.db"
    study_name  = "all_search"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize", #minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(min_resource=30)
    )

    N_TRIALS = 2000
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        logger.info("Interrupted by user...")

    # Summarize results
    logger.info("Study Complete. Best Trial:")
    logger.info(f"  Value: {study.best_trial.value:.4f}")
    logger.info("  Params:")
    for k, v in study.best_trial.params.items():
        logger.info(f"    {k}: {v}")

    from optuna.trial import TrialState
    pruned_trials   = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")

    logger.info("Done! You can now run:")
    logger.info("  optuna-dashboard sqlite:///all_search.db")
    logger.info("Then open http://127.0.0.1:8080 in your browser to monitor the study progress.")
