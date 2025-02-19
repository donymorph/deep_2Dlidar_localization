import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
class GraphLocalizationNet(nn.Module):
    """
    Graph-LSTM Network for LiDAR Localization.
    
    This model processes a sequence of LiDAR scans. Each scan (a 1D tensor with 'num_beams' range values)
    is first converted into a graph where:
      - Each node has features [range, angle] (angle is linearly spaced from -pi to pi).
      - Edges connect each node with its immediate neighbors (bidirectional).
      
    The graph is processed with several GCNConv layers to obtain a graph-level embedding. Then, an LSTM
    processes the sequence of embeddings to capture temporal dynamics, and a final fully connected layer outputs
    the predicted pose: [pos_x, pos_y, orientation_z].
    
    Args:
        num_beams (int): Number of LiDAR beams per scan (e.g. 360).
        in_channels (int): Number of features per node (default: 2 for [range, angle]).
        hidden_channels (int): Hidden dimension size for GCN layers.
        gcn_layers (int): Number of GCNConv layers to use.
        lstm_hidden (int): Hidden size of the LSTM.
        output_size (int): Number of output dimensions (typically 3: x, y, z).
        dropout (float): Dropout probability applied in both GCN and LSTM blocks.
    """
    def __init__(self, input_size=360, in_channels=2, hidden_channels=64, gcn_layers=3, 
                 lstm_hidden=192, output_size=3, dropout=0.0):
        super(GraphLocalizationNet, self).__init__()
        self.num_beams = input_size
        self.in_channels = in_channels
        
        # Define GCN layers that process a single LiDAR scan graph.
        self.convs = nn.ModuleList()
        # First GCN layer: maps node features from in_channels to hidden_channels.
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Additional GCN layers.
        for _ in range(gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_dropout = dropout
        
        # Fully connected layer to refine graph-level embedding from GCN.
        self.fc_graph = nn.Linear(hidden_channels, hidden_channels)
        
        # LSTM to model temporal dependencies over a sequence of scan embeddings.
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=lstm_hidden, 
                            num_layers=3, batch_first=True, dropout=dropout)
        
        # Fully connected output layer for final pose regression.
        self.fc_out = nn.Linear(lstm_hidden, output_size)
    
    def convert_scan_to_graph_embedding(self, scan):
        """
        Converts a single LiDAR scan (shape: [batch_size, num_beams]) into a graph-level embedding.
        
        For each scan:
          - Node features: [range, angle] where angle is linearly spaced from -pi to pi.
          - Edges: Each node is connected to its immediate neighbor (bidirectional).
          
        Returns:
            Tensor of shape [batch_size, hidden_channels] representing the scan embedding.
        """
        batch_size, num_beams = scan.shape
        # Create an angles vector for all beams.
        angles = torch.linspace(-torch.pi, torch.pi, num_beams, device=scan.device)
        data_list = []
        for i in range(batch_size):
            scan_i = scan[i]  # shape: [num_beams]
            # Stack range and angle to create node features: [num_beams, 2]
            x = torch.stack([scan_i, angles], dim=1)
            
            # Create edges: connect node j to node j+1 (bidirectional)
            edge_index = []
            for j in range(num_beams - 1):
                edge_index.append([j, j + 1])
                edge_index.append([j + 1, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=scan.device).t().contiguous()
            
            data_obj = Data(x=x, edge_index=edge_index)
            data_list.append(data_obj)
        
        batch_data = Batch.from_data_list(data_list)
        
        # Process the graph with GCN layers.
        x, edge_index, batch_idx = batch_data.x, batch_data.edge_index, batch_data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.gcn_dropout, training=self.training)
        # Global mean pooling to aggregate node features to a graph-level embedding.
        graph_emb = global_mean_pool(x, batch_idx)
        # Refine the embedding via a fully connected layer.
        graph_emb = F.relu(self.fc_graph(graph_emb))
        return graph_emb

    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data (Tensor): Raw input tensor of shape [batch_size, seq_len, num_beams]
                           representing a sequence of LiDAR scans.
        Returns:
            Tensor: Output predictions of shape [batch_size, output_size]
        """
        if len(data.shape) == 2:
            data = data.unsqueeze(1)  # Add sequence dimension if missing

        # data is expected to be of shape [batch_size, seq_len, num_beams]
        batch_size, seq_len, num_beams = data.shape
        
        # Process each time step separately to get scan-level embeddings.
        embeddings = []
        for t in range(seq_len):
            scan = data[:, t, :]  # Shape: [batch_size, num_beams]
            emb_t = self.convert_scan_to_graph_embedding(scan)  # Shape: [batch_size, hidden_channels]
            embeddings.append(emb_t.unsqueeze(1))  # Add time dimension
        
        # Concatenate embeddings along the time dimension.
        embeddings = torch.cat(embeddings, dim=1)  # Shape: [batch_size, seq_len, hidden_channels]
        
        # Feed the sequence of embeddings into the LSTM.
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        # Use the output at the last time step.
        out = self.fc_out(lstm_out[:, -1, :])
        return out
##############################################
# test functions for the GraphLocalizationNet
#############################################

def generate_synthetic_lidar_data2(batch_size=1, num_beams=360, room_size=8.0):
    """
    Simulates 2D LiDAR scans in a square room environment.
    """
    torch.manual_seed(42)  
    angles = torch.linspace(-np.pi, np.pi, num_beams)  
    lidar_scans = torch.zeros(batch_size, num_beams)

    for i in range(batch_size):
        distances = torch.full((num_beams,), room_size)
        for j, angle in enumerate(angles):
            x_dist = room_size / abs(np.cos(angle)) if np.cos(angle) != 0 else room_size
            y_dist = room_size / abs(np.sin(angle)) if np.sin(angle) != 0 else room_size
            distances[j] = min(x_dist, y_dist)
        lidar_scans[i] = distances

    return lidar_scans

def visualize_graph(graph_data, num_nodes=360):
    """
    Visualizes the graph representation of a LiDAR scan.
    
    Args:
        graph_data (torch_geometric.data.Data): The graph data to visualize.
        num_nodes (int): Number of LiDAR beams (nodes).
    """
    angles = graph_data.x[:, 1].cpu().numpy()  
    ranges = graph_data.x[:, 0].cpu().numpy()

    # Convert polar coordinates (range, angle) to Cartesian (x, y)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    edge_index = graph_data.edge_index.cpu().numpy()

    # Create Graph using NetworkX
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, pos=(x[i], y[i]))  # Node positions
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        G.add_edge(src, dst)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(7, 7))
    nx.draw(G, pos, node_size=10, edge_color='gray', alpha=0.6, with_labels=False)
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Graph Representation of LiDAR Scan")
    plt.grid(True)
    plt.show()

    
def visualize_predictions(predictions, ground_truth=None):
    """
    Plots model predictions (x, y) vs. ground truth locations.
    """
    plt.figure(figsize=(6, 6))
    
    # Ground truth (optional)
    if ground_truth is not None:
        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], c='green', label="Ground Truth")
    
    # Predicted positions
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', marker='x', label="Predictions")
    
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Model Predictions vs. Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_lidar_batch1d(lidar_batch):
    """
    Visualizes a batch of LiDAR scans.
    
    Args:
        lidar_batch (Tensor): Tensor of shape [batch_size, num_beams] representing a batch of LiDAR scans.
    """
    batch_size, num_beams = lidar_batch.shape
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=batch_size, ncols=1, figsize=(10, 2 * batch_size))
    
    if batch_size == 1:
        axes = [axes]  # Ensure axes is iterable for a single batch
    
    for i in range(batch_size):
        scan = lidar_batch[i].cpu().numpy()
        angles = np.linspace(-np.pi, np.pi, num_beams)
        
        axes[i].plot(angles, scan)
        axes[i].set_title(f"LiDAR Scan {i+1}")
        axes[i].set_xlabel("Angle (radians)")
        axes[i].set_ylabel("Distance")
    
    plt.tight_layout()
    plt.show()
    
def visualize_lidar_batch2d(lidar_batch):
    """
    Visualizes a batch of LiDAR scans in Cartesian coordinates.
    
    Args:
        lidar_batch (Tensor): Tensor of shape [batch_size, num_beams] representing a batch of LiDAR scans.
    """
    batch_size, num_beams = lidar_batch.shape
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=batch_size, ncols=1, figsize=(10, 2 * batch_size))
    
    if batch_size == 1:
        axes = [axes]  # Ensure axes is iterable for a single batch
    
    for i in range(batch_size):
        scan = lidar_batch[i].cpu().numpy()
        angles = np.linspace(-np.pi, np.pi, num_beams)
        
        # Convert polar to Cartesian coordinates
        x = scan * np.cos(angles)
        y = scan * np.sin(angles)
        
        axes[i].scatter(x, y, s=1)
        axes[i].set_title(f"LiDAR Scan {i+1}")
        axes[i].set_xlabel("X (meters)")
        axes[i].set_ylabel("Y (meters)")
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Generate synthetic sequential LiDAR data that imitates real scans in a square room.
    def generate_synthetic_lidar_sequence(batch_size=4, seq_len=10, num_beams=360, room_size=8.0):
        """
        Generates a sequence of LiDAR scans.
        For simplicity, each scan in the sequence is identical (simulating a static scene).
        """
        torch.manual_seed(42)
        angles = torch.linspace(-torch.pi, torch.pi, num_beams)
        scan = torch.zeros(num_beams)
        for j, angle in enumerate(angles):
            x_dist = room_size / abs(np.cos(angle)) if np.cos(angle) != 0 else room_size
            y_dist = room_size / abs(np.sin(angle)) if np.sin(angle) != 0 else room_size
            scan[j] = min(x_dist, y_dist)
        # Repeat the same scan over the sequence and batch.
        lidar_seq = scan.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
        return lidar_seq

    # Generate synthetic sequential LiDAR data.
    batch_size = 4
    seq_len = 10
    num_beams = 360
    synthetic_sequence = generate_synthetic_lidar_sequence(batch_size, seq_len, num_beams)

    # Create an instance of the model.
    model = GraphLocalizationNet(input_size=num_beams, in_channels=2, hidden_channels=64, 
                                       gcn_layers=3, lstm_hidden=192, output_size=3, dropout=0.3)
    # Run a forward pass.
    predictions = model(synthetic_sequence)
    print("Predictions shape:", predictions.shape)  # Expected: [batch_size, 3]
    print("Predictions (first sample):", predictions[0].detach().numpy())

    # Visualization: Show one scan's graph representation.
    # We'll use the conversion function from a single time step.
    gln = GraphLocalizationNet(input_size=num_beams, in_channels=2, hidden_channels=64, 
                                     gcn_layers=3, lstm_hidden=192, output_size=3, dropout=0.3)
    # Use the first time step of the first sample.
    single_scan = synthetic_sequence[0, 0, :]  # Shape: [num_beams]
    # Convert to graph data.
    angles = torch.linspace(-torch.pi, torch.pi, num_beams)
    x = torch.stack([single_scan, angles], dim=1)  # [num_beams, 2]
    edge_index = []
    for j in range(num_beams - 1):
        edge_index.append([j, j+1])
        edge_index.append([j+1, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    from torch_geometric.data import Data
    data_obj = Data(x=x, edge_index=edge_index)
    
    # For visualization, we use NetworkX to plot the graph.
    import networkx as nx
    pos = {}
    x_np = x[:, 0].detach().numpy()
    angles_np = x[:, 1].detach().numpy()
    # Convert polar to Cartesian for visualization.
    xs = x_np * np.cos(angles_np)
    ys = x_np * np.sin(angles_np)
    for i in range(num_beams):
        pos[i] = (xs[i], ys[i])
    G = nx.Graph()
    for i in range(num_beams):
        G.add_node(i, pos=pos[i])
    edge_index_np = edge_index.numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        G.add_edge(src, dst)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_size=10, edge_color='gray', with_labels=False)
    plt.title("Graph Representation of One LiDAR Scan")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()