import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerRegressor(nn.Module):
    """
    Minimal transformer-based regressor example in PyTorch.

    For LiDAR scans, you might treat each beam as a token. This example
    uses the standard nn.TransformerEncoder to process an input sequence
    and aggregates the final representation for regression.

    - input_size: dimension of each token (e.g., 1 if each beam is scalar range)
    - n_tokens: length of the sequence (number of LiDAR beams)
    - d_model: dimension inside the transformer
    - nhead: number of attention heads
    - num_layers: number of TransformerEncoder layers
    - output_size: e.g., 6 for linear + angular velocities
    """
    def __init__(self, n_tokens, data_dim=1, d_model=64, nhead=8,
                 num_layers=2, output_size=3):
        super().__init__()
        self.n_tokens = n_tokens
        self.input_size = data_dim
        self.d_model = d_model

        # 1) Input linear projection from input_size -> d_model
        self.input_projection = nn.Linear(data_dim, d_model)

        # 2) Positional encoding (basic version)
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_tokens, d_model))

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Final output layer
        self.regressor = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x shape: (batch, n_tokens) if input_size=1
                 or (batch, n_tokens, input_size) if input_size>1

        We'll unify shapes to (batch, n_tokens, input_size).
        Then transform to (n_tokens, batch, d_model) for the PyTorch Transformer.
        """
        if x.dim() == 2:
            # (batch, n_tokens) -> (batch, n_tokens, 1)
            x = x.unsqueeze(-1)
        batch_size, seq_len, in_dim = x.shape

        # Project to d_model
        x_proj = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x_proj = x_proj + self.positional_encoding[:, :seq_len, :]

        # Transformer needs shape (seq_len, batch, d_model)
        x_proj = x_proj.permute(1, 0, 2)  # (seq_len, batch, d_model)

        # Pass through encoder
        encoded = self.transformer_encoder(x_proj)  # (seq_len, batch, d_model)

        # For regression, we can take the mean or the final token
        # Here, let's just take the "mean" across seq_len dimension
        encoded_mean = encoded.mean(dim=0)  # (batch, d_model)

        out = self.regressor(encoded_mean)  # (batch, output_size)
        return out

# ---------------------------

###############################################################################
# 4) 1D CNN with Transformer /// tested works a bit heavier computationally expensive
###############################################################################
class ConvTransformerNet(nn.Module):
    """
    Combines Conv1D and Transformer layers for processing 1D LiDAR scans and predicting robot position.
    """
    def __init__(self, input_size=360, output_size=3, d_model=256, nhead=4, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(ConvTransformerNet, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv = nn.Sequential(
            DepthwiseSeparableConv1D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            DepthwiseSeparableConv1D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            DepthwiseSeparableConv1D(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            DepthwiseSeparableConv1D(in_channels=64, out_channels=d_model, kernel_size=3, stride=2, padding=2, dilation=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        # Positional encoding
        self.positional_encoding = PositionalEncodingCov(d_model=d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensures input is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        # Input shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)

        # Pass through Conv1D layers
        out = self.conv(x)  # Shape: (batch_size, d_model, seq_len)

        # Reshape for Transformer (batch, seq_len, d_model)
        out = out.permute(0, 2, 1)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        out = self.positional_encoding(out)

        # Pass through Transformer Encoder
        out = self.transformer_encoder(out)  # Shape: (batch, seq_len, d_model)

        # Use the output of the last token
        out = out[:, -1, :]  # Shape: (batch, d_model)

        # Pass through fully connected layers
        out = self.fc(out)  # Shape: (batch, output_size)
        return out
    
class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, dilation=dilation)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class PositionalEncodingCov(nn.Module):
    """
    Adds positional encoding to the input sequence to allow the Transformer to capture order information.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingCov, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)  # Sequence length from input
        return x + self.encoding[:, :seq_len, :].to(x.device)  # Add positional encoding
    
class CNNTransformerNet_Optuna(nn.Module):
    """
    A hybrid 1D CNN + Transformer for LiDAR-based regression tasks.
    Batch Size: 16
    Learning Rate: 6.89e-5
    Optimizer: Adam
    """
    def __init__(self, 
                 input_size=360,
                 activation_fn=nn.Tanh, 
                 cnn_dropout=0.0, 
                 d_model=256, 
                 nhead=4, 
                 num_transformer_layers=4, 
                 transformer_dropout=0.1, 
                 output_size=3):
        super().__init__()
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            # Conv Layer 0
            nn.Conv1d(1, 48, kernel_size=7, stride=1, padding=7 // 2),
            nn.BatchNorm1d(48),
            activation_fn(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Conv Layer 1
            nn.Conv1d(48, 48, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(48),
            activation_fn(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Conv Layer 2
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=5 // 2),
            nn.BatchNorm1d(64),
            activation_fn(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Conv Layer 3
            nn.Conv1d(64, 24, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(24),
            activation_fn(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Flatten dimension
        dummy_input = torch.zeros(1, 1, 360)  # Assuming input size = 360
        with torch.no_grad():
            cnn_out = self.conv_layers(dummy_input)
        self.cnn_output_dim = cnn_out.view(1, -1).size(1)

        # Linear embedding layer to map CNN output to d_model
        self.embed_fc = nn.Linear(self.cnn_output_dim, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final regression output layer
        self.final_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        Forward pass through the CNN + Transformer architecture.
        """
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:  # If missing sequence dimension
            x = x.unsqueeze(1)  # Add dummy sequence dimension

        batch_size, seq_len, input_dim = x.shape

        # Reshape to (batch*seq_len, 1, input_dim) for CNN
        x = x.view(batch_size * seq_len, 1, input_dim)
        cnn_out = self.conv_layers(x)  # Output: (batch*seq_len, num_channels, reduced_seq_len)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Flatten to (batch*seq_len, cnn_output_dim)

        # Embed to d_model
        embed_out = self.embed_fc(cnn_out)  # Output: (batch*seq_len, d_model)

        # Reshape back to (batch, seq_len, d_model) for Transformer
        embed_out = embed_out.view(batch_size, seq_len, -1)

        # Pass through Transformer
        transformer_out = self.transformer(embed_out)  # Output: (batch, seq_len, d_model)

        # Take the last token output for regression
        final_out = transformer_out[:, -1, :]  # Last time step
        return self.final_fc(final_out)
    
    
# Positional Encoding (sine-cosine)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=360):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(0)])
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# LiDARFormer: Hybrid Transformer + CNN architecture for LiDAR localization.
class LiDARFormer(nn.Module):
    def __init__(self, input_size, output_size, in_features=2, d_model=64, 
                 num_transformer_layers=3, num_cnn_layers=2, cnn_kernel_size=3, dropout=0.1):
        """
        Args:
          input_size: number of LiDAR beams (e.g., 360)
          output_size: number of regression outputs (3: pos_x, pos_y, orientation_z)
          in_features: number of features per beam (2 for 2D LiDAR)
          d_model: embedding dimension
          num_transformer_layers: number of Transformer encoder layers (global context)
          num_cnn_layers: number of convolutional layers in the CNN branch (local features)
          cnn_kernel_size: kernel size for CNN branch
          dropout: dropout rate applied in Transformer and CNN branches
        """
        super(LiDARFormer, self).__init__()
        self.input_size = input_size  # expected number of beams
        self.in_features = in_features
        
        # 1. Embedding and Positional Encoding.
        self.embedding = nn.Linear(in_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=input_size)
        
        # 2. Transformer Branch (global context).
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 3. CNN Branch (local features).
        # Expect input to CNN branch to be (B, d_model, input_size).
        cnn_layers = []
        in_channels = d_model
        for i in range(num_cnn_layers):
            cnn_layers.append(nn.Conv1d(in_channels, d_model, kernel_size=cnn_kernel_size, 
                                        stride=1, padding=cnn_kernel_size // 2))
            cnn_layers.append(nn.BatchNorm1d(d_model))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = d_model
        self.cnn_branch = nn.Sequential(*cnn_layers)
        
        # Compute the CNN branch output dimension.
        dummy_cnn = torch.zeros(1, d_model, input_size)
        with torch.no_grad():
            cnn_out = self.cnn_branch(dummy_cnn)
        cnn_out_dim = cnn_out.size(2)  # reduced sequence length after pooling
        
        # 4. Fusion: We'll fuse the Transformer (global) and CNN (local) features.
        # For the CNN branch, we upsample (via interpolation) its output back to input_size length.
        self.fuse_conv = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.fuse_bn = nn.BatchNorm1d(d_model)
        self.fuse_relu = nn.ReLU()
        
        # 5. Pooling and Output.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        """
        Args:
          x: input tensor of shape (B, input_size) or (B, input_size, in_features)
          If a 2D tensor is provided, it is assumed to have shape (B, input_size) and will be expanded.
        Returns:
          Tensor of shape (B, output_size)
        """
        # Ensure x has shape (B, input_size, in_features)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, input_size, 1)
            if self.in_features == 2:
                x = x.repeat(1, 1, 2)  # (B, input_size, 2)
        
        # 1. Embedding and Positional Encoding.
        x_embed = self.embedding(x)            # (B, input_size, d_model)
        x_embed = self.pos_enc(x_embed)          # (B, input_size, d_model)
        
        # 2. Transformer Branch.
        # The transformer expects input shape (B, seq_len, d_model)
        global_features = self.transformer_encoder(x_embed)  # (B, input_size, d_model)
        
        # 3. CNN Branch.
        # Rearrange to (B, d_model, input_size)
        cnn_input = x_embed.transpose(1, 2)      # (B, d_model, input_size)
        local_features = self.cnn_branch(cnn_input)  # (B, d_model, reduced_seq)
        # Upsample local features back to the original sequence length.
        local_features = nn.functional.interpolate(local_features, size=self.input_size, mode='linear', align_corners=False)
        # Rearrange back to (B, input_size, d_model)
        local_features = local_features.transpose(1, 2)
        
        # 4. Fusion.
        # Concatenate global and local features along the feature dimension.
        fused = torch.cat([global_features, local_features], dim=2)  # (B, input_size, 2*d_model)
        # Rearrange for 1D convolution fusion: (B, 2*d_model, input_size)
        fused = fused.transpose(1, 2)
        fused = self.fuse_conv(fused)   # (B, d_model, input_size)
        fused = self.fuse_bn(fused)
        fused = self.fuse_relu(fused)
        # Rearrange back to (B, input_size, d_model)
        fused = fused.transpose(1, 2)
        
        # 5. Pooling and Output.
        # Global average pooling over the beam (sequence) dimension.
        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        out = self.fc(pooled)  # (B, output_size)
        return out
    
