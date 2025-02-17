import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba  # Import the Mamba block from mamba-ssm

# Positional encoding to inject beam-order information.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=360):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# Simple Mamba architecture.
class MambaModel_simple(nn.Module):
    def __init__(self, input_size, output_size, in_features=2, d_model=32,
                 num_blocks=4, dropout=0.1, d_state=16, d_conv=4, expand=2):
        """
        Args:
          input_size: number of LiDAR beams (e.g., 360)
          in_features: number of features per beam (2 for 2d lidar)
          d_model: embedding dimension
          num_blocks: number of stacked Mamba blocks
          dropout: dropout rate after each block
          d_state, d_conv, expand: hyperparameters for the Mamba block
          output_size: number of regression outputs (3: pos_x, pos_y, orientation_z)
        """
        super(MambaModel_simple, self).__init__()
        # Embed each beam (with 2 features) into a d_model vector.
        self.embedding = nn.Linear(in_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=input_size)
        
        # Stack Mamba blocks with residual connections.
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                nn.Dropout(dropout)
            ) for _ in range(num_blocks)
        ])
        
        # Pool over the sequence dimension.
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Final fully connected layer.
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Expected x shape: (batch_size, input_size, in_features)
        # If input is 2D (batch_size, input_size), add and duplicate feature dimension.
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # -> (B, input_size, 1)
            x = x.repeat(1, 1, 2)  # -> (B, input_size, 2)
        
        x = self.embedding(x)         # -> (B, input_size, d_model)
        x = self.pos_enc(x)           # Add positional encoding
        
        # Apply each block with a residual connection.
        for block in self.blocks:
            x = x + block(x)
        
        # Pool along the sequence dimension.
        x = x.transpose(1, 2)         # -> (B, d_model, input_size)
        x = self.pool(x).squeeze(-1)   # -> (B, d_model)
        out = self.fc(x)              # -> (B, output_size)
        return out

# Simple Mamba2 architecture.
class Mamba2Model(nn.Module):
    def __init__(self, input_size, output_size, in_features=2, d_model=32,
                 num_blocks=4, dropout=0.1, d_state=64, d_conv=4, expand=2):
        """
        Uses Mamba2 blocks (imported from mamba_ssm) with a larger default state (d_state=64).
        """
        super(Mamba2Model, self).__init__()
        self.embedding = nn.Linear(in_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=input_size)
        
        from mamba_ssm import Mamba2  # Import Mamba2 block from mamba_ssm
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                nn.Dropout(dropout)
            ) for _ in range(num_blocks)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: expected shape (batch_size, input_size, in_features)
        # If input is 2D, add the feature dimension and duplicate to have 2 channels.
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, input_size, 1)
            x = x.repeat(1, 1, 2)  # (B, input_size, 2)
        
        # Apply embedding and positional encoding.
        x = self.embedding(x)   # -> (B, input_size, d_model)
        x = self.pos_enc(x)     # -> (B, input_size, d_model)
        
        # Force a channels_last layout by temporarily unsqueezing to 4D.
        # Our desired shape is (B, input_size, d_model), so we add a dummy channel dimension.
        x = x.unsqueeze(1)  # (B, 1, input_size, d_model)
        x = x.contiguous(memory_format=torch.channels_last)
        x = x.squeeze(1)    # (B, input_size, d_model)
        
        # Process through each Mamba2 block with residual connection.
        for block in self.blocks:
            # Ensure the input to each block has the desired layout.
            xb = x.unsqueeze(1)  # (B, 1, input_size, d_model)
            xb = xb.contiguous(memory_format=torch.channels_last)
            xb = xb.squeeze(1)    # (B, input_size, d_model)
            x = x + block(xb)
        
        # Pool along the sequence dimension.
        x = x.transpose(1, 2)         # -> (B, d_model, input_size)
        x = self.pool(x).squeeze(-1)   # -> (B, d_model)
        out = self.fc(x)              # -> (B, output_size)
        return out
    
# PositionalEncoding adds sine–cosine positional embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=360):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(0)])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# Hybrid CNN-Mamba network for LiDAR localization.
class MambaModel(nn.Module):
    def __init__(self, input_size, output_size, in_features=2, d_model=64, 
                 num_cnn_channels=32, num_mamba_blocks=3, dropout=0.1):
        """
        Args:
          input_size: number of LiDAR beams (e.g., 360)
          output_size: regression outputs (3: pos_x, pos_y, orientation_z)
          in_features: number of features per beam (2 for 2D LiDAR)
          d_model: embedding dimension
          num_cnn_channels: base number of channels for multi-scale CNN branch
          num_mamba_blocks: number of Mamba blocks in the global branch
          dropout: dropout rate for both branches
        """
        super(MambaModel, self).__init__()
        self.input_size = input_size  # e.g. 360
        self.in_features = in_features  # e.g. 2
        self.d_model = d_model
        
        # 1. Embedding & Positional Encoding
        # Embed each beam (2 features) into a d_model-dimensional vector.
        self.embedding = nn.Linear(in_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=input_size)
        
        # 2. CNN Branch: Multi-scale convolutions.
        # Input to the CNN branch is (B, d_model, input_size). We use three parallel conv layers.
        self.cnn_branch = nn.ModuleDict({
            'conv3': nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=num_cnn_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(num_cnn_channels),
                nn.ReLU()
            ),
            'conv5': nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=num_cnn_channels, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(num_cnn_channels),
                nn.ReLU()
            ),
            'conv7': nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=num_cnn_channels, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(num_cnn_channels),
                nn.ReLU()
            )
        })
        # Fuse the multi-scale outputs: concatenate channels and use a 1x1 convolution to map back to d_model.
        self.cnn_fuse = nn.Sequential(
            nn.Conv1d(in_channels=num_cnn_channels * 3, out_channels=d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 3. Global Branch: Mamba blocks for long-range modeling.
        # This branch takes the embedded sequence (B, input_size, d_model) and processes it.
        self.mamba_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                nn.Dropout(dropout)
            ) for _ in range(num_mamba_blocks)
        ])
        
        # 4. Fusion: Combine CNN branch (local features) and Mamba branch (global context).
        # We concatenate along the feature dimension and then fuse.
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU()
        )
        
        # 5. Pooling & Regression Head.
        # Global average pooling over the beam dimension then fully connected layer.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        """
        x: (B, input_size, in_features)
        If a 2D tensor (B, input_size) is provided, we assume a single feature per beam and duplicate it.
        """
        # If input is 2D, reshape to (B, input_size, 1) and repeat to have 2 features.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            if self.in_features == 2:
                x = x.repeat(1, 1, 2)
        
        # 1. Embedding and Positional Encoding.
        x_embed = self.embedding(x)         # (B, input_size, d_model)
        x_embed = self.pos_enc(x_embed)       # (B, input_size, d_model)
        
        # 2. CNN Branch.
        # Rearrange to (B, d_model, input_size) for conv1d.
        cnn_input = x_embed.transpose(1, 2)
        out_conv3 = self.cnn_branch['conv3'](cnn_input)
        out_conv5 = self.cnn_branch['conv5'](cnn_input)
        out_conv7 = self.cnn_branch['conv7'](cnn_input)
        cnn_concat = torch.cat([out_conv3, out_conv5, out_conv7], dim=1)  # (B, num_cnn_channels*3, input_size)
        cnn_fused = self.cnn_fuse(cnn_concat)                              # (B, d_model, input_size)
        cnn_features = cnn_fused.transpose(1, 2)                           # (B, input_size, d_model)
        
        # 3. Global Branch.
        mamba_features = self.mamba_blocks(x_embed)  # (B, input_size, d_model)
        
        # 4. Fusion.
        fused = torch.cat([cnn_features, mamba_features], dim=2)  # (B, input_size, 2*d_model)
        fused = self.fuse(fused)                                  # (B, input_size, d_model)
        
        # 5. Pooling and Output.
        # Transpose to (B, d_model, input_size), pool, then flatten.
        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)      # (B, d_model)
        out = self.fc(pooled)                                     # (B, output_size)
        return out
