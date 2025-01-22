import os
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import LidarOdomDataset
from splitting import split_dataset
import torch.nn.functional as F

########################################
# Logging setup
########################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(console)

class CNNLSTMNet(nn.Module):
    """
    Flexible 1D CNN + LSTM for LiDAR scans over time.
    If seq_len=1 (no temporal data), the LSTM won't exploit time info.
    """
    def __init__(
        self,
        conv_params_list,   # list of dicts, each describing one conv layer
        activation_fn,      # e.g. nn.ReLU
        cnn_dropout=0.0,    # dropout after each conv block
        hidden_size=128,
        lstm_num_layers=1,
        lstm_dropout=0.0,   # dropout in LSTM
        output_size=3
    ):
        """
        conv_params_list: 
            [
              {"out_channels":..., "kernel_size":..., "pool":True/False, "stride":..., ...},
              ...
            ]
        For each conv layer, we do: Conv1d -> BN -> activation -> dropout? -> optional Pool
        Then flatten -> LSTM -> final FC.
        """
        super().__init__()
        
        conv_layers = []
        in_ch = 1  # input shape = (batch, 1, input_size)
        for cdict in conv_params_list:
            out_ch   = cdict["out_channels"]
            ksize    = cdict["kernel_size"]
            stride   = cdict.get("stride", 1)
            pool     = cdict.get("pool", False)
            # pad
            padding  = ksize // 2

            conv_layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                         kernel_size=ksize, stride=stride, padding=padding))
            conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(activation_fn())
            if cnn_dropout > 1e-6:
                conv_layers.append(nn.Dropout(cnn_dropout))
            if pool:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)

        # Figure out the flatten dimension
        dummy = torch.zeros(1,1,360)
        with torch.no_grad():
            dummy_out = self.conv(dummy)   # shape (1, outC, new_len)
        conv_flat_dim = dummy_out.view(1,-1).size(1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_flat_dim,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape can be (batch, seq_len, input_size) or (batch, input_size).
        If no seq_len dimension, we add it.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        b, seq, inp_size = x.shape

        # Reshape to (b*seq, 1, input_size) for CNN
        x = x.view(b*seq, 1, inp_size)
        feat = self.conv(x)  # shape: (b*seq, outC, new_len)
        feat = feat.view(feat.size(0), -1)  # (b*seq, conv_flat_dim)

        # Reshape to (b, seq, conv_flat_dim) for LSTM
        feat = feat.view(b, seq, -1)
        out, (h, c) = self.lstm(feat)
        # out: (b, seq, hidden_size)
        final = out[:, -1, :]  # last time step
        return self.fc(final)


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
def train_and_eval(model, train_loader, val_loader, device, trial, max_epochs=10):
    """
    Train model for up to max_epochs, each epoch do:
      - forward/backward on train
      - measure val_loss
      - call trial.report(val_loss, epoch)
      - if trial.should_prune(): raise TrialPruned
    """
    criterion = nn.HuberLoss()

    # Choose LR
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # Choose optimizer
    opt_name = trial.suggest_categorical("optimizer", ["Adam","AdamW","RMSprop","SGD"])
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
        # train
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

        # val
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for lidar_batch, odom_batch in val_loader:
                lidar_batch = lidar_batch.to(device).float()
                odom_batch  = odom_batch.to(device).float()
                out  = model(lidar_batch)
                val_loss += criterion(out, odom_batch).item()
        val_loss /= len(val_loader)

        # report to Optuna
        trial.report(val_loss, epoch)
        # check pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        model.train()

    return val_loss

########################################
# 4) Objective
########################################
def objective(trial):
    # We recreate train_loader, val_loader with the chosen batch_size
    batch_size = trial.suggest_categorical("batch_size", [16,32,64])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    # Number of conv layers
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
    conv_params_list = []
    for i in range(num_conv_layers):
        out_ch   = trial.suggest_int(f"conv{i}_out_channels", 8, 64, step=8)
        ksize    = trial.suggest_int(f"conv{i}_kernel_size", 3, 7, step=2)
        stride   = trial.suggest_int(f"conv{i}_stride", 1, 2)
        pool     = trial.suggest_categorical(f"conv{i}_pool", [False, True])
        conv_params_list.append({
            "out_channels": out_ch,
            "kernel_size":  ksize,
            "stride":       stride,
            "pool":         pool
        })
        # Activation
    activation_name = trial.suggest_categorical("activation_fn", ["ReLU","Tanh","LeakyReLU"])
    activation_fn   = getattr(nn, activation_name)
    
    # CNN dropout
    cnn_dropout = trial.suggest_float("cnn_dropout", 0.0, 0.5, step=0.1)

    # LSTM params
    hidden_size = trial.suggest_int("lstm_hidden_size", 32, 256, step=32)
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    lstm_dropout    = trial.suggest_float("lstm_dropout", 0.0, 0.5, step=0.1)

    # Build the model
    model = CNNLSTMNet(
        conv_params_list=conv_params_list,
        activation_fn=activation_fn,
        cnn_dropout=cnn_dropout,
        hidden_size=hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        output_size=3
    ).to(device)
    # partial training with pruning
    
    max_epochs = 10
    val_loss   = train_and_eval(model, train_loader, val_loader, device, trial, max_epochs)
    return val_loss

########################################
# Main
########################################
if __name__ == "__main__":
    # Create dataset
    train_ds, val_ds, test_ds, input_size, output_size = create_dataset_and_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # We'll store references so objective can see them
    globals()['train_ds'] = train_ds
    globals()['val_ds']   = val_ds
    globals()['test_ds']  = test_ds
    globals()['input_size'] = input_size
    globals()['output_size'] = output_size
    globals()['device']      = device

    # Create or load study from SQLite, enabling pruner
    storage_url = "sqlite:///cnn_lstm.db"
    study_name  = "cnn_lstm_search"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),          # or whichever sampler
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)  # e.g. median pruner
    )

    N_TRIALS = 10
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        logger.info("Interrupted by user...")

    # Summarize
    logger.info("Study Complete. Best Trial:")
    logger.info(f"  Value: {study.best_trial.value:.4f}")
    logger.info("  Params:")
    for k,v in study.best_trial.params.items():
        logger.info(f"    {k}: {v}")

    # Show pruned vs. completed
    from optuna.trial import TrialState
    pruned_trials   = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")

    logger.info("Done! Now you can run:")
    logger.info("  optuna-dashboard sqlite:///cnn_lstm.db")
    logger.info("Then open http://127.0.0.1:8080 in your browser to see the study progress.")
