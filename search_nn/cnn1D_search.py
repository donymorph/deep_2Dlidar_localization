import os
import math
import csv
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import LidarOdomDataset
from splitting import split_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(console)

########################################
# 1) CNN 1D Model Class
########################################
class Conv1DNet(nn.Module):
    """
    A flexible 1D CNN with optional dropout in conv or FC layers.
    """
    def __init__(
        self,
        input_size,
        output_size,
        conv_params_list,
        fc_hidden_size,
        activation_fn,
        conv_dropout=0.0,
        fc_dropout=0.0
    ):
        super().__init__()
        layers = []
        in_channels = 1  # assume (batch,1,input_size)
        for i, cdict in enumerate(conv_params_list):
            out_ch   = cdict["out_channels"]
            ksize    = cdict["kernel_size"]
            stride   = cdict["stride"]
            pool     = cdict["pool"]

            padding = ksize // 2
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=ksize,
                stride=stride,
                padding=padding
            )
            layers.append(conv)
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(activation_fn())
            if conv_dropout > 0.0:
                layers.append(nn.Dropout(conv_dropout))

            if pool:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_channels = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # figure out flatten dim
        dummy = torch.zeros(1,1,input_size)
        dummy_out = self.conv_layers(dummy)
        flatten_dim = dummy_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, fc_hidden_size),
            activation_fn(),
            nn.Dropout(fc_dropout) if fc_dropout>0 else nn.Identity(),
            nn.Linear(fc_hidden_size, output_size)
        )

    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # (batch,1,input_size)
        c = self.conv_layers(x)
        c = c.view(c.size(0), -1)
        out = self.fc(c)
        return out

########################################
# 2) Utility: DataLoader creation
########################################
def create_data_loaders():
    dataset = LidarOdomDataset("dataset/odom_data.csv","dataset/scan_data.csv")
    train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.2, 0.1)
    # We'll sample batch size from the trial, but we set a default here
    # We'll override in the objective
    train_loader = DataLoader(train_ds,  batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,    batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=32, shuffle=False)

    input_size  = len(dataset[0][0])
    output_size = len(dataset[0][1])
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_size, output_size

########################################
# 3) Train/Eval with partial epochs
########################################
def train_and_eval(model, train_loader, val_loader, device, trial, max_epochs=10):
    """
    multiple epochs, after each epoch we measure val_loss
    and call `trial.report(val_loss, epoch)`. If `trial.should_prune()`, we prune.
    """
    criterion = nn.HuberLoss()
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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

        # Evaluate on val
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for lidar_batch, odom_batch in val_loader:
                lidar_batch = lidar_batch.to(device).float()
                odom_batch  = odom_batch.to(device).float()
                preds = model(lidar_batch)
                val_loss += criterion(preds, odom_batch).item()
        val_loss /= len(val_loader)

        # report to optuna
        trial.report(val_loss, epoch)
        # pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        model.train()

    return val_loss

########################################
# 4) Objective (with pruner + new search space)
########################################
def objective(trial):
    # 4.1) Sample batch size from e.g. [16,32,64]
    global train_ds, val_ds, test_ds
    batch_size = trial.suggest_categorical("batch_size", [16,32,64])
    # re-create train_loader, val_loader with new batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4.2) CNN architecture
    num_conv_layers = trial.suggest_int("num_conv_layers",1,4)
    conv_params_list = []
    for i in range(num_conv_layers):
        out_ch = trial.suggest_int(f"conv{i}_out_channels", 8, 64, step=8)
        ksize  = trial.suggest_int(f"conv{i}_kernel_size", 3,7, step=2)
        stride = trial.suggest_int(f"conv{i}_stride", 1,2)
        pool   = trial.suggest_categorical(f"conv{i}_pool", [False, True])

        conv_params_list.append({
            "out_channels": out_ch,
            "kernel_size":  ksize,
            "stride":       stride,
            "pool":         pool
        })

    fc_hidden_size = trial.suggest_int("fc_hidden_size", 32,256, step=32)
    activation_name = trial.suggest_categorical("activation_fn", ["ReLU","Tanh","LeakyReLU"])
    activation_fn   = getattr(nn, activation_name)

    conv_dropout = trial.suggest_float("conv_dropout", 0.0, 0.5, step=0.1)
    fc_dropout   = trial.suggest_float("fc_dropout",   0.0, 0.5, step=0.1)

    # Build the model
    model = Conv1DNet(
        input_size=input_size,
        output_size=output_size,
        conv_params_list=conv_params_list,
        fc_hidden_size=fc_hidden_size,
        activation_fn=activation_fn,
        conv_dropout=conv_dropout,
        fc_dropout=fc_dropout
    ).to(device)

    # We'll do up to 10 epochs with pruner
    max_epochs = 10
    val_loss   = train_and_eval(model, train_loader, val_loader, device, trial, max_epochs=max_epochs)
    return val_loss

########################################
# Main
########################################
if __name__=="__main__":
    # 1) Create data
    train_ds, val_ds, test_ds, train_loader0, val_loader0, test_loader, input_size, output_size = create_data_loaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device}")

    # We'll store references to the DS for objective
    globals()['train_ds'] = train_ds
    globals()['val_ds']   = val_ds
    globals()['test_ds']  = test_ds
    globals()['input_size'] = input_size
    globals()['output_size'] = output_size
    globals()['device'] = device

    # 2) Optuna Study w/ pruning + logging to SQLite
    storage_url = "sqlite:///cnn1D_search.db"
    study_name  = "Conv1D_search"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),           # e.g. TPE
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2) # simple median pruner
    )

    N_TRIALS = 50
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        logger.info("Interrupted by user...")

    # Summarize
    logger.info("Study Complete. Best Trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info("  Params:")
    for k,v in study.best_trial.params.items():
        logger.info(f"    {k}: {v}")

    # Show pruned vs. completed
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of completed trials: {len(complete_trials)}")

    logger.info("Done! Now you can run:")
    logger.info("  optuna-dashboard sqlite:///cnn1D_search.db")
    logger.info("Then open http://127.0.0.1:8080 in your browser to see the study progress.")