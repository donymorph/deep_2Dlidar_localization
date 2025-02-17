import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMNet_modified(nn.Module):
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
            num_layers=5,
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
    
    
    
class Attention(nn.Module):
    """
    Simple attention mechanism for LSTM outputs.
    Computes attention weights for each time step and returns
    the weighted sum (context vector) as the output.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)
        attn_scores = self.attn(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)
        return context

class CNNLSTMNet_modified2(nn.Module):
    """
    Enhanced CNN-LSTM network for predicting position (x, y) and orientation (z)
    from 1D LiDAR scan data. This version adds more Conv1d layers for richer feature extraction.
    
    Architecture Summary:
      - Expanded CNN block with 6 convolutional layers (each with BatchNorm, Tanh, Dropout, and pooling).
      - A 5-layer LSTM with dropout and an attention mechanism over the LSTM outputs.
      - A two-layer fully connected head with dropout and Tanh activation for the final prediction.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # =====================
        # Extended Convolutional Feature Extractor
        # =====================
        self.conv_layers = nn.Sequential(
            # Conv Layer 0
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=5 // 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            #nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Conv Layer 1
            nn.Conv1d(in_channels=32, out_channels=48, kernel_size=5, stride=2, padding=5 // 2),
            nn.BatchNorm1d(48),
            nn.Tanh(),
            #nn.Dropout(p=0.3),
            
            # Conv Layer 2
            nn.Conv1d(in_channels=48, out_channels=32, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            #nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Conv Layer 3
            nn.Conv1d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(48),
            nn.Tanh(),
            #nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # New Conv Layer 4
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            #nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # New Conv Layer 5
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            #nn.Dropout(p=0.1)
            # (Optional: Add MaxPool1d here if further downsampling is desired)
        )
        
        # Compute the flattened dimension after CNN layers using a dummy input.
        dummy_input = torch.zeros(1, 1, input_size)
        with torch.no_grad():
            conv_out = self.conv_layers(dummy_input)  # Shape: (batch, channels, seq_length)
        conv_flat_dim = conv_out.size(1) * conv_out.size(2)
        
        # =====================
        # LSTM for Temporal Modeling
        # =====================
        self.lstm = nn.LSTM(
            input_size=conv_flat_dim,
            hidden_size=192,
            num_layers=1,
            batch_first=True,
            #dropout=0.5  # Applies dropout between LSTM layers
        )
        
        # =====================
        # Attention Mechanism over LSTM Outputs
        # =====================
        self.attention = Attention(hidden_size=192)
        
        # =====================
        # Fully Connected Head for Final Prediction
        # =====================
        self.fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
                              If single-scan inputs, seq_len can be 1.
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size).
        """
        # If input is 2D (batch, input_size), add a sequence dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Now shape: (batch, 1, input_size)
        b, seq, inp_size = x.shape

        # Reshape to process each scan with CNN: (batch * seq, 1, input_size)
        x = x.view(b * seq, 1, inp_size)
        x = self.conv_layers(x)  # Apply CNN layers
        # Reshape for LSTM: (batch, seq, conv_flat_dim)
        x = x.view(b, seq, -1)
        
        # LSTM processing: lstm_out has shape (batch, seq, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism to obtain a context vector (batch, hidden_size)
        context = self.attention(lstm_out)
        
        # Pass the context vector through the fully connected head for final prediction
        output = self.fc(context)
        return output