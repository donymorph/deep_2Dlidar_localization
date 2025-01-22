import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##############################################################################
# 1. Feature extractor for 360 LiDAR
##############################################################################

class LiDARFeatureExtractor(nn.Module):
    """
    A more advanced aggregator than a simple MLP:
    - 1D Convolutional layers or a small Transformer, etc.
    Here, we do a small 1D CNN with multiple kernels and residual blocks.
    """
    def __init__(self, input_size=360, latent_dim=64):
        super().__init__()
        # input shape: (batch, 1, 360) if we treat LiDAR as 1D
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)

        # final linear to get latent_dim
        self.fc = nn.Linear(64 * (input_size // 8), latent_dim)

    def forward(self, x):
        # x shape: (batch, 360)
        # reshape to (batch, 1, 360)
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))  # (batch,16,180)
        x = F.relu(self.conv2(x))  # (batch,32,90)
        x = F.relu(self.conv3(x))  # (batch,64,45)

        # flatten
        x = x.view(x.size(0), -1)  # shape (batch, 64*45) if input_size=360
        x = self.fc(x)            # shape (batch, latent_dim)
        return x


##############################################################################
# 2. SINDy-inspired block that constructs a library of candidate functions
##############################################################################

class SINDyLibrary(nn.Module):
    """
    Creates a library of candidate functions from the combined state vector.
    For example, polynomials up to certain order, trigonometric terms, etc.
    
    The input state can be: [z (latent features), x, y, yaw].
    We'll build a library of terms like:
      1, x, y, yaw, z_i, x^2, y^2, yaw^2, x*y, ...
      sin(x), cos(yaw), ...
    Then we multiply by a learnable sparse matrix Xi to get the derivatives or next state.
    """
    def __init__(self, in_dim, poly_order=2, include_trig=True):
        """
        in_dim: dimension of [z, x, y, yaw]
        poly_order: polynomial order
        include_trig: whether to include sin/cos of each input dimension
        """
        super().__init__()
        self.in_dim = in_dim
        self.poly_order = poly_order
        self.include_trig = include_trig

    def forward(self, state):
        """
        state shape: (batch, in_dim)
        returns a library shape: (batch, n_terms)
        """
        # Start library with constant 1
        library = [torch.ones_like(state[:, 0:1])]

        # Add linear terms
        library.append(state)  # each dimension

        # Add polynomial terms up to poly_order
        if self.poly_order >= 2:
            # x_i * x_j for i <= j
            for i in range(self.in_dim):
                for j in range(i, self.in_dim):
                    library.append(state[:, i:i+1] * state[:, j:j+1])

        # Add trig terms if desired
        if self.include_trig:
            library.append(torch.sin(state))
            library.append(torch.cos(state))

        # Concatenate
        return torch.cat(library, dim=1)  # shape: (batch, n_terms)


##############################################################################
# 3. Main model: feature extractor + SINDy library + linear layer (sparse)
##############################################################################

class PhysicsInformedSINDyModel(nn.Module):
    """
    Full model:
      1) Extract LiDAR features -> z
      2) Combine z with (x, y, yaw) to build a 'state'
      3) Build library of candidate functions (SINDy style)
      4) Multiply by a linear layer to get [dx, dy, dyaw]
      5) Return (x_next, y_next, yaw_next) = (x, y, yaw) + dt * (dx, dy, dyaw)
    """
    def __init__(self,
                 lidar_input_dim=360,
                 latent_dim=64,
                 poly_order=2,
                 include_trig=True,
                 output_size=3):
        """
        For consistency with your training loop:
         - output_size=3 => we predict (x, y, yaw).
        """
        super().__init__()

        self.feature_extractor = LiDARFeatureExtractor(
            input_size=lidar_input_dim, 
            latent_dim=latent_dim
        )

        # The combined "state" dimension = latent_dim + 3 (for x, y, yaw)
        self.state_dim = latent_dim + 3

        self.sindy_library = SINDyLibrary(
            in_dim=self.state_dim, 
            poly_order=poly_order, 
            include_trig=include_trig
        )

        # We'll create a linear layer from the library dimension -> 3
        # (predicting [dx, dy, dyaw]).
        # We pick a maximum possible library size for 2nd-order + trig.
        max_library_dim = (
            1                               # constant
            + self.state_dim                # linear
            + (self.state_dim*(self.state_dim+1))//2  # pairwise products
            + (2 if include_trig else 0) * self.state_dim  # sin & cos
        )
        self.sindy_linear = nn.Linear(max_library_dim, 3, bias=False)

        # Example time step
        self.dt = 0.1

    def forward(self, lidar_in, odom_in):
        """
        lidar_in:  (batch, 360)  => LiDAR features
        odom_in:   (batch, 3)    => current (x, y, yaw)
        
        Returns:
          preds: (batch, 3) => (x_next, y_next, yaw_next)
        """
        # Unpack current states
        x_in = odom_in[:, 0]
        y_in = odom_in[:, 1]
        yaw_in = odom_in[:, 2]

        # 1) Extract LiDAR features => z
        z = self.feature_extractor(lidar_in)  # (batch, latent_dim)

        # 2) Combine z with (x_in, y_in, yaw_in)
        # shape => (batch, latent_dim + 3)
        state_t = torch.cat([z, x_in.unsqueeze(1), y_in.unsqueeze(1), yaw_in.unsqueeze(1)], dim=1)

        # 3) SINDy library
        library_t = self.sindy_library(state_t)  # (batch, n_terms)

        # 4) Predict derivatives => [dx, dy, dyaw]
        dxyz = self.sindy_linear(library_t)  # (batch, 3)

        # 5) Next states
        x_next = x_in + self.dt * dxyz[:, 0]
        y_next = y_in + self.dt * dxyz[:, 1]
        yaw_next = yaw_in + self.dt * dxyz[:, 2]

        # Return them as shape (batch, 3)
        preds = torch.stack([x_next, y_next, yaw_next], dim=1)
        return preds
