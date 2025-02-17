import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class GraphLocalizationNet(nn.Module):
    """
    Graph Neural Network for LiDAR Localization.
    
    The model treats each LiDAR beam as a node with features (e.g., range, angle).
    It applies several GCNConv layers to aggregate local spatial features,
    then uses global mean pooling to obtain a graph-level representation.
    Finally, a fully connected MLP maps the graph embedding to the target output:
    position (x, y) and orientation (z).
    
    This updated version can accept either a torch_geometric.data.Data object or a
    raw tensor of shape [batch_size, num_beams]. When given a tensor, it converts each
    LiDAR scan into a graph where:
      - Node features: [range, angle] (angle computed from -pi to pi)
      - Edges: A simple chain connection (each node connected to its neighbors)
    
    Args:
        in_channels (int): Number of features per node (default 2: [range, angle]).
        hidden_channels (int): Hidden dimension size for GCN layers.
        output_size (int): Number of output dimensions (typically 3: x, y, z).
        num_layers (int): Number of GCNConv layers.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, input_size=360, in_channels=2, hidden_channels=64, output_size=3, num_layers=5, dropout=0.0):
        super(GraphLocalizationNet, self).__init__()
        self.convs = nn.ModuleList()
        # First GCN layer: from in_channels to hidden_channels.
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Additional GCN layers.
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        # Fully connected layers for graph-level regression.
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_size)
    
    def convert_tensor_to_graph(self, tensor_data):
        """
        Converts a tensor of shape [batch_size, num_beams] into a batched
        torch_geometric.data.Batch object where each LiDAR scan is represented as a graph.
        
        For each scan:
          - Node features: [range, angle] where angle is linearly spaced from -pi to pi.
          - Edge index: Connect each node with its immediate neighbors (bidirectional).
        """
        batch_size, num_beams = tensor_data.shape
        # Create angles vector: shape [num_beams]
        angles = torch.linspace(-torch.pi, torch.pi, num_beams, device=tensor_data.device)
        data_list = []
        for i in range(batch_size):
            scan = tensor_data[i]  # shape: [num_beams]
            # Create node features: each node gets [range, angle]
            x = torch.stack([scan, angles], dim=1)  # shape: [num_beams, 2]
            # Create edge_index: simple chain connections
            edge_index = []
            for j in range(num_beams):
                if j < num_beams - 1:
                    edge_index.append([j, j+1])
                    edge_index.append([j+1, j])
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=tensor_data.device).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=tensor_data.device)
            data_obj = Data(x=x, edge_index=edge_index)
            data_list.append(data_obj)
        batch_data = Batch.from_data_list(data_list)
        return batch_data

    def forward(self, data):
        """
        Forward pass.
        
        If 'data' is a raw tensor, convert it to a graph data object.
        
        Args:
            data (torch_geometric.data.Data or Tensor): If tensor, shape [batch_size, num_beams]
        
        Returns:
            torch.Tensor: Output predictions of shape [batch_size, output_size]
        """
        # If data is a tensor, convert it to a Batch object.
        if isinstance(data, torch.Tensor):
            data = self.convert_tensor_to_graph(data)
        
        # Now data is assumed to be a Data or Batch object with attributes x, edge_index, batch.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Pass node features through GCN layers.
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Global pooling: aggregate node embeddings into a single graph-level embedding.
        x = global_mean_pool(x, batch)
        # Fully connected MLP for final regression.
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc2(x)
        return out

# Example usage:
if __name__ == "__main__":
    # For demonstration, create a dummy tensor of shape [batch_size, num_beams]
    batch_size = 16
    num_beams = 360
    dummy_tensor = torch.randn(batch_size, num_beams)
    
    # Create an instance of the network.
    model = GraphLocalizationNet(in_channels=2, hidden_channels=64, output_size=3, num_layers=3, dropout=0.3)
    # Forward pass using raw tensor (conversion happens inside the forward method)
    prediction = model(dummy_tensor)
    print("Prediction shape:", prediction.shape)  # Expected: [16, 3]
