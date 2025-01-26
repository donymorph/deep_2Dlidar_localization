import os
import math
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import LidarOdomDataset
from splitting import split_dataset

########################################
# Logging setup
########################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(console)

########################################
# 1) MLP Model Class (with dropout)
########################################
class MLP(nn.Module):
    """
    Basic MLP with optional dropout after each hidden layer.
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation_fn, dropout=0.0):
        super().__init__()
        layers = []
        prev_size = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(activation_fn())
            if dropout > 1e-6:
                layers.append(nn.Dropout(dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
def train_and_eval_mlp(model, train_loader, val_loader, device, trial, max_epochs=10):
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

    # hidden layers
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    hidden_sizes = []
    for i in range(num_hidden_layers):
        h_size = trial.suggest_int(f"hidden_layer_{i}_size", 32, 256, step=32)
        hidden_sizes.append(h_size)

    # activation
    activation_name = trial.suggest_categorical("activation_fn", ["ReLU","Tanh","LeakyReLU"])
    activation_fn = getattr(nn, activation_name)

    # dropout
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    # build model
    model = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        activation_fn=activation_fn,
        dropout=dropout
    ).to(device)

    # partial training with pruning
    max_epochs = 100
    val_loss   = train_and_eval_mlp(model, train_loader, val_loader, device, trial, max_epochs)
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
    storage_url = "sqlite:///mlp_search.db"
    study_name  = "MLP_search"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),          # or whichever sampler
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)  # e.g. median pruner
    )

    N_TRIALS = 500
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
    logger.info("  optuna-dashboard sqlite:///mlp_search.db")
    logger.info("Then open http://127.0.0.1:8080 in your browser to see the study progress.")
