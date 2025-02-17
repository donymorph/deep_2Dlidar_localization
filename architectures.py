# architectures.py
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1) MLP Variants //// tested works
###############################################################################
class SimpleMLP(nn.Module):
    """
    A simple feed-forward network: Input -> FC(128) -> ReLU -> FC(64) -> ReLU -> FC(6).
    """
    def __init__(self, input_size, hidden1=128, hidden2=64, output_size=3):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x):
        return self.net(x)


class MLP_Optuna(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with architecture defined by Optuna hyperparameter tuning:
    - 3 hidden layers with sizes [256, 256, 224]
    - LeakyReLU activation
    - No dropout
    - Adam optimizer and learning rate 0.000137
    - huberloss: 0.1670 at 100 epochs
    - batch_size: 16
    """
    def __init__(self, input_size, output_size):
        super(MLP_Optuna, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),   # Hidden Layer 0
            nn.LeakyReLU(),
            nn.Linear(256, 256),         # Hidden Layer 1
            nn.LeakyReLU(),
            nn.Linear(256, 224),         # Hidden Layer 2
            nn.LeakyReLU(),
            nn.Linear(224, output_size)  # Output Layer
        )

    def forward(self, x):
        return self.net(x)

###############################################################################
# 3) 1D CNN  ///////// tested works
###############################################################################
class Conv1DNet(nn.Module):
    """
    Example of using 1D convolution for LiDAR scans (treated like 1D signals).
    Note: You must handle input shape accordingly.
    Output_size = [(Input_size - kernel_size + 2 * padding)/stride] + 1
    """
    def __init__(self, input_size, output_size=3):
        super(Conv1DNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.Linear(128 * 91, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        out = self.conv(x)  # (batch_size, 128, 91)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out

class Conv1DNet_Optuna(nn.Module):
    """
    Optimized 1D CNN model based on the best trial results from Optuna.
    num_conv_layers: 4
    batch_size: 16
    learning rate : 0.0019218642787654772
    optimizer: Adam
    Convolutional layers dropout: 0.0.
    Fully connected layers dropout: 0.0.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Layer 0
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            # Layer 1
            nn.Conv1d(16, 40, kernel_size=5, stride=2, padding=5 // 2),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv1d(40, 56, kernel_size=5, stride=2, padding=5 // 2),
            nn.BatchNorm1d(56),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv1d(56, 56, kernel_size=5, stride=1, padding=5 // 2),
            nn.BatchNorm1d(56),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Compute flatten dimension using a dummy tensor
        dummy = torch.zeros(1, 1, input_size)
        flatten_dim = self.conv_layers(dummy).view(1, -1).size(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch, 1, input_size)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
###############################################################################
# 4) 1D CNN with LSTM /////// tested works
###############################################################################
class Conv1DLSTMNet(nn.Module):
    """
    Combines Conv1D and LSTM layers for processing 1D LiDAR scans and predicting robot position.
    """
    def __init__(self, input_size=360, output_size=3, hidden_size=64, lstm_layers=2):
        super(Conv1DLSTMNet, self).__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,  # Match the out_channels from Conv1D
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # Reshape to (batch, 1, input_size)
        out = self.conv(x)  # Conv output: (batch, 128, 91)

        # Reshape for LSTM: (batch, seq_len, features)
        out = out.permute(0, 2, 1)  # Change to (batch, seq_len=91, features=128)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(out)  # lstm_out: (batch, seq_len, hidden_size)

        # Use the last hidden state (h_n) as the summary of the sequence
        lstm_out = lstm_out[:, -1, :]  # Extract the last time step: (batch, hidden_size)

        # Fully connected layers
        out = self.fc(lstm_out)
        return out
class CNNLSTMNet_Optuna(nn.Module):
    """
    Optimized CNN-LSTM network based on best trial results.
    Combines convolutional layers for feature extraction and LSTM layers for temporal modeling.
    batch_size: 16
    lr: 0.0006829381720401536
    optimizer: Adam
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            # Conv Layer 0
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=5 // 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Conv Layer 1
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=5 // 2),
            nn.BatchNorm1d(48),
            nn.Tanh(),
            # Conv Layer 2
            nn.Conv1d(48, 32, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Conv Layer 3
            nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(48),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Flatten dimension computation
        dummy_input = torch.zeros(1, 1, input_size)
        with torch.no_grad():
            conv_out = self.conv_layers(dummy_input)  # Output shape: (batch, channels, seq_length)
        conv_flat_dim = conv_out.size(1) * conv_out.size(2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=conv_flat_dim,
            hidden_size=192,
            num_layers=1,
            batch_first=True,
            #dropout=0.4
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(192, output_size)

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Shape (batch, seq_len, input_size).
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension (batch, 1, input_size)
        b, seq, inp_size = x.shape

        # Reshape for CNN (batch * seq_len, 1, input_size)
        x = x.view(b * seq, 1, inp_size)
        x = self.conv_layers(x)  # Apply CNN layers
        x = x.view(b, seq, -1)   # Reshape for LSTM (batch, seq_len, conv_flat_dim)

        # Pass through LSTM
        lstm_out, (h, c) = self.lstm(x)
        final_output = lstm_out[:, -1, :]  # Take the last time step's output

        # Fully connected output layer
        return self.fc(final_output)

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
        self.positional_encoding = PositionalEncoding(d_model=d_model)

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
    
class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input sequence to allow the Transformer to capture order information.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
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


###############################################################################
# 5) Transformer itself /// working
###############################################################################
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


###############################################################################
# 6) LSTM / RNN /// havent tested yet //chatgpt generated
###############################################################################
class LSTMRegressor(nn.Module):
    """
    A simple LSTM-based regressor.

    - input_size: dimension of each time step input
    - hidden_size: size of LSTM hidden state
    - num_layers: number of stacked LSTM layers
    - output_size: dimension of regression output
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # LSTM init hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))  
        # lstm_out shape: (batch, seq_len, hidden_size)

        # We can take the output at the final time step
        out = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(out)        # (batch, output_size)
        return out


class RNNRegressor(nn.Module):
    """
    A simple vanilla RNN regressor.

    - input_size: dimension of each time step input
    - hidden_size: size of RNN hidden state
    - num_layers: stacked layers
    - output_size: dimension of regression output
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        rnn_out, hn = self.rnn(x, h0)
        out = rnn_out[:, -1, :]  # final timestep
        out = self.fc(out)
        return out



###############################################################################
# 8) GAN (Generic Skeleton) /////// buggy havent tested //chatgpt generated
###############################################################################
class SimpleGenerator(nn.Module):
    """
    Example generator for a GAN, typically used for generating data, not regression.
    """
    def __init__(self, latent_dim=128, output_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class SimpleDiscriminator(nn.Module):
    """
    Example discriminator for a GAN.
    """
    def __init__(self, input_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


###############################################################################
# 7) SNN (Spiking Neural Network) - Minimal Placeholder ///////// buggy havent tested //chatgpt generated
###############################################################################
try:
    import norse.torch as snn
    # If you're using Norse or another SNN library, you can implement more advanced code.
    # Below is a simple placeholder using Norse library.
    class SimpleSpikingNet(nn.Module):
        """
        Minimal spiking neural network example using Norse.
        This is purely illustrative and may not run without the correct library.
        """
        def __init__(self, input_size=1, hidden_size=64, output_size=6):
            super().__init__()
            # A single-layer spiking linear
            self.slayer = snn.LIFRecurrentCell(input_size, hidden_size)
            self.readout = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            """
            x shape: (time, batch, input_size) if treating each time step, or adapt as needed.
            """
            s = torch.zeros(x.size(1), self.slayer.state_size, device=x.device)
            outputs = []
            for t in range(x.size(0)):
                z, s = self.slayer(x[t], s)
                outputs.append(z)
            # final state -> readout
            out = self.readout(outputs[-1])
            return out

except ImportError:
    class SimpleSpikingNet(nn.Module):
        """
        Fallback dummy SNN if Norse or other spiking library is not installed.
        """
        def __init__(self, input_size=1, hidden_size=64, output_size=6):
            super().__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, x):
            # Not truly spiking. Just pass through a linear layer.
            return self.fc(x)


###############################################################################
# 8) KNN (Conceptual Placeholder)      ///////// buggy havent tested //chatgpt generated
###############################################################################
class KNNRegressor:
    """
    Placeholder for a K-Nearest Neighbors Regressor in PyTorch style.
    Typically, you'd use scikit-learn's KNeighborsRegressor or similar.
    """
    def __init__(self, k=5):
        self.k = k
        self.x_data = None
        self.y_data = None

    def fit(self, x, y):
        """
        Store training data for nearest-neighbor lookups.
        x: (N, input_size)
        y: (N, output_size)
        """
        self.x_data = x
        self.y_data = y

    def predict(self, x_query):
        """
        For each query in x_query, find the k nearest points in self.x_data,
        average their y_data as the prediction.
        """
        if self.x_data is None or self.y_data is None:
            raise ValueError("KNNRegressor has not been fitted with data.")
        
        # Basic L2 distance
        # x_query shape: (M, input_size)
        # x_data shape: (N, input_size)
        # We'll compute distances for each query to each data point
        x_query_expanded = x_query.unsqueeze(1)   # (M, 1, input_size)
        x_data_expanded = self.x_data.unsqueeze(0)  # (1, N, input_size)
        dist = (x_query_expanded - x_data_expanded).pow(2).sum(dim=2)  # (M, N)

        # For each row in dist, find top-k smallest
        _, idx = torch.topk(dist, k=self.k, largest=False, dim=1)  # (M, k)

        # Gather y_data
        neighbors_y = self.y_data[idx]  # shape: (M, k, output_size)
        prediction = neighbors_y.mean(dim=1)  # (M, output_size)
        return prediction

    def forward(self, x_query):
        """
        For PyTorch style usage, define forward as an alias for predict.
        """
        return self.predict(x_query)
