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

########################################
# 1) 1D CNN + Transformer Model Class (with dropout)
########################################
class CNNTransformerNet(nn.Module):
    """
    Extended 1D CNN + Transformer approach with multiple conv layers, optional dropout.
    """
    def __init__(
        self,
        conv_params_list,  # list of conv layer configs
        activation_fn,
        cnn_dropout=0.0,
        d_model=64,
        nhead=4,
        num_transformer_layers=2,
        transformer_dropout=0.1,
        output_size=3
    ):
        super().__init__()
        conv_layers = []
        in_ch = 1
        for i, cdict in enumerate(conv_params_list):
            out_ch = cdict["out_channels"]
            ksize  = cdict["kernel_size"]
            stride = cdict["stride"]
            pool   = cdict["pool"]
            padding = ksize // 2

            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=ksize,
                                         stride=stride, padding=padding))
            conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(activation_fn())
            if cnn_dropout>1e-6:
                conv_layers.append(nn.Dropout(cnn_dropout))
            if pool:
                conv_layers.append(nn.MaxPool1d(2,2))

            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)

        # Flatten dimension
        dummy = torch.zeros(1,1,360)
        with torch.no_grad():
            dummy_out = self.conv(dummy)  # shape (1, out_ch, new_len)
        conv_dim = dummy_out.view(1,-1).size(1)

        self.d_model = d_model
        self.embed_fc = nn.Linear(conv_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.final_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size) or (batch, input_size)
        if missing seq dimension, we add it.
        We'll CNN each time step => flatten => embed => transformer => final
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch,1,input_size)
        b, seq, inp_size = x.shape

        # reshape => (b*seq,1,inp_size)
        x = x.view(b*seq, 1, inp_size)
        feat = self.conv(x)  # shape (b*seq, out_ch, new_len)
        feat = feat.view(feat.size(0), -1)  # (b*seq, conv_dim)

        # embed => (b*seq, d_model)
        feat = self.embed_fc(feat)
        # reshape => (b, seq, d_model)
        feat = feat.view(b, seq, self.d_model)

        # pass through transformer
        out = self.transformer(feat)  # shape (b, seq, d_model)
        final = out[:, -1, :]  # last time step
        return self.final_fc(final)


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
    # 1) We recreate train_loader, val_loader with the chosen batch_size
    batch_size = trial.suggest_categorical("batch_size", [16,32,64])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 2) conv layers
    num_conv_layers = trial.suggest_int("num_conv_layers",1,4)
    conv_params_list = []
    for i in range(num_conv_layers):
        out_ch = trial.suggest_int(f"conv{i}_out_channels",8,64,step=8)
        ksize  = trial.suggest_int(f"conv{i}_kernel_size",3,7,step=2)
        stride = trial.suggest_int(f"conv{i}_stride",1,2)
        pool   = trial.suggest_categorical(f"conv{i}_pool",[False,True])
        conv_params_list.append({
            "out_channels": out_ch,
            "kernel_size":  ksize,
            "stride":       stride,
            "pool":         pool
        })

    # 3) activation
    activation_name = trial.suggest_categorical("activation_fn",["ReLU","LeakyReLU","Tanh"])
    activation_fn   = getattr(nn, activation_name)

    # dropout in CNN
    cnn_dropout = trial.suggest_float("cnn_dropout",0.0,0.5, step=0.1)

    # 4) Transformer
    d_model      = trial.suggest_int("d_model",32,256,step=32)
    valid_nheads = [h for h in [1,2,4,8] if d_model % h == 0]
    nhead = trial.suggest_categorical("nhead", valid_nheads)

    if d_model % nhead != 0:
        # We skip or prune this trial, so it doesn't cause an error
        raise optuna.exceptions.TrialPruned(f"d_model={d_model} not divisible by nhead={nhead}.")
    num_tf_layers= trial.suggest_int("num_tf_layers",1,4)
    tf_dropout   = trial.suggest_float("transformer_dropout",0.0,0.3,step=0.1)

    # build
    model = CNNTransformerNet(
        conv_params_list=conv_params_list,
        activation_fn=activation_fn,
        cnn_dropout=cnn_dropout,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_tf_layers,
        transformer_dropout=tf_dropout,
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
    storage_url = "sqlite:///cnn_transformer.db"
    study_name  = "cnn_transformer_search"

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
    logger.info("  optuna-dashboard sqlite:///cnn_transformer.db")
    logger.info("Then open http://127.0.0.1:8080 in your browser to see the study progress.")
