import torch
import torch.nn as nn
import torch.nn.functional as F

def knn_query(x, k=8, downsample_factor=4):
    """
    KNN-based downsampling function. Finds k-nearest neighbors and downsamples the input.

    Args:
        x: (batch_size, input_size) - Input LiDAR data.
        k: Number of neighbors to aggregate.
        downsample_factor: Factor by which the input size should be reduced.

    Returns:
        downsampled_features: (batch_size, input_size // downsample_factor)
    """
    batch_size, input_size = x.shape
    downsampled_size = input_size // downsample_factor  # Compute target downsample size

    # Compute pairwise Euclidean distance between beams
    distance = torch.cdist(x.unsqueeze(-1), x.unsqueeze(-1), p=2).squeeze(-1)  # (batch, input_size, input_size)

    # Find k-nearest neighbors indices
    knn_indices = distance.topk(k, largest=False, dim=-1)[1]  # (batch, input_size, k)

    # Gather features from nearest neighbors
    knn_features = torch.gather(x.unsqueeze(1).expand(-1, input_size, -1), 2, knn_indices)  # (batch, input_size, k)

    # Mean pooling over k neighbors
    aggregated_features = knn_features.mean(dim=-1)  # (batch, input_size)

    # Proper Downsampling: Take every Nth feature for dimensionality reduction
    downsampled_features = aggregated_features[:, :downsampled_size * downsample_factor:downsample_factor]

    return downsampled_features

class PointFeaturePyramid(nn.Module):
    """
    Extracts hierarchical spatial features using KNN, MLPs with max-pooling, and Tanh activation.
    """
    def __init__(self, input_size=90, feature_dim=128):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(90, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)  # Global max-pooling

    def forward(self, x):
        """
        x: (batch_size, input_size)
        """
        x = knn_query(x)  # Aggregate features using KNN (batch_size, input_size)
        x = self.mlp1(x)  # (batch_size, 128)
        #x = self.maxpool(x.unsqueeze(1)).squeeze(1)  # Max-pooling to reduce noise
        x = torch.tanh(self.mlp2(x))  # (batch_size, feature_dim)
        return x

class PeepholeLSTM(nn.Module):
    """
    Temporal Feature Propagation using Peephole LSTM.
    Explicitly connects cell state to input, forget, and output gates.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.peephole_wf = nn.Parameter(torch.randn(hidden_dim))  # Peephole connection to Forget gate
        self.peephole_wi = nn.Parameter(torch.randn(hidden_dim))  # Peephole connection to Input gate
        self.peephole_wo = nn.Parameter(torch.randn(hidden_dim))  # Peephole connection to Output gate

    def forward(self, x):
        """
        x: (batch_size, seq_len, feature_dim)
        """
        out, (h_n, c_n) = self.lstm(x)

        # Peephole connection applied directly to gates
        forget_gate = torch.sigmoid(out + self.peephole_wf * c_n[-1])  # (batch, seq_len, hidden_dim)
        input_gate = torch.sigmoid(out + self.peephole_wi * c_n[-1])
        output_gate = torch.sigmoid(out + self.peephole_wo * c_n[-1])

        cell_state = input_gate * torch.tanh(c_n[-1]) + forget_gate * c_n[-1]
        last_hidden = output_gate * torch.tanh(cell_state)  # (batch_size, seq_len, hidden_dim)
        
        return last_hidden[:, -1, :]  # Return last timestep output (batch_size, hidden_dim)

class GatedPoseRefinement(nn.Module):
    """
    Gated Hierarchical Pose Refinement with GRU-like updates.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.update_gate = nn.Linear(hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pose = nn.Linear(hidden_dim, 3)  # Output: (pos_x, pos_y, orientation_z)

    def forward(self, x):
        """
        x: (batch_size, hidden_dim)
        """
        z = torch.sigmoid(self.update_gate(x))  # Update gate
        r = torch.sigmoid(self.reset_gate(x))  # Reset gate
        h_new = r * x  # Element-wise scaling

        refined_pose = self.fc_pose(h_new * z)  # Predict final pose
        return refined_pose

class DSLOModel(nn.Module):
    """
    Deep Sequence LiDAR Odometry (DSLO) with:
    - KNN Query
    - Peephole LSTM
    - Gated Pose Refinement
    """
    def __init__(self, input_size=360, output_size=3, feature_dim=128, hidden_dim=192):
        super().__init__()
        self.feature_extractor = PointFeaturePyramid(input_size, feature_dim)
        self.temporal_propagation = PeepholeLSTM(feature_dim, hidden_dim)
        self.pose_refinement = GatedPoseRefinement(hidden_dim)

    def forward(self, lidar_batch):
        """
        lidar_batch: (batch_size, 360) -> LiDAR scans
        """
        # 1) Extract spatial features
        spatial_features = self.feature_extractor(lidar_batch)  # (batch, feature_dim)

        # 2) Temporal Propagation with Peephole LSTM
        temporal_features = self.temporal_propagation(spatial_features.unsqueeze(1))  # Add sequence dim

        # 3) Gated Pose Refinement
        refined_pose = self.pose_refinement(temporal_features)  # (batch, 3)
        
        return refined_pose
