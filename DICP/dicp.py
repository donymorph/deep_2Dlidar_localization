import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# (Re)Use the vec2mat, DifferentiableICP, and LocalizationNet definitions
# from your architecture code.

def vec2mat(trans):
    """
    Converts a batch of 2D transformations from vector to homogeneous matrix form.
    trans: Tensor of shape (batch, 3) where each row is [theta, tx, ty].
    Returns: Tensor of shape (batch, 3, 3)
    """
    batch_size = trans.shape[0]
    cos_theta = torch.cos(trans[:, 0])
    sin_theta = torch.sin(trans[:, 0])
    tx = trans[:, 1]
    ty = trans[:, 2]
    
    T = torch.zeros((batch_size, 3, 3), device=trans.device, dtype=trans.dtype)
    T[:, 0, 0] = cos_theta
    T[:, 0, 1] = -sin_theta
    T[:, 0, 2] = tx
    T[:, 1, 0] = sin_theta
    T[:, 1, 1] = cos_theta
    T[:, 1, 2] = ty
    T[:, 2, 2] = 1.0
    return T

class DifferentiableICP(nn.Module):
    """
    Advanced differentiable ICP module that accumulates transformation updates.
    Given an initial transformation (batch, 3) in the format [theta, tx, ty],
    it iteratively refines the alignment between a source and target 2D point cloud
    by computing small delta updates and composing them via homogeneous matrices.
    """
    def __init__(self, iterations=5):
        super(DifferentiableICP, self).__init__()
        self.iterations = iterations

    def forward(self, source, target, init_transformation):
        source = source.float()
        target = target.float()
        init_transformation = init_transformation.float()
        # source and target: (batch, N, 2)
        # init_transformation: (batch, 3) where each row is [theta, tx, ty]
        batch_size = init_transformation.shape[0]
        
        # Convert the initial transformation to homogeneous matrix form.
        T = vec2mat(init_transformation)  # (batch, 3, 3)
        
        # Convert source point cloud to homogeneous coordinates.
        N = source.shape[1]
        ones = torch.ones((batch_size, N, 1), device=source.device, dtype=source.dtype)
        source_h = torch.cat([source, ones], dim=2)  # (batch, N, 3)
        
        # Apply the current cumulative transformation T to the source.
        source_transformed_h = torch.bmm(source_h, T.transpose(1, 2))
        source_transformed = source_transformed_h[:, :, :2]  # (batch, N, 2)
        
        for i in range(self.iterations):
            # Compute pairwise squared Euclidean distances between transformed source and target.
            diff = source_transformed.unsqueeze(2) - target.unsqueeze(1)
            dists = torch.sum(diff ** 2, dim=-1)  # (batch, N, N)
            
            # Soft assignment: each source point gets a soft correspondence over target points.
            weights = nn.functional.softmax(-dists, dim=-1)  # (batch, N, N)
            
            # Compute soft correspondences for each source point.
            target_corr = torch.bmm(weights, target)  # (batch, N, 2)
            
            # Compute centroids of the current source and its correspondences.
            centroid_source = torch.mean(source_transformed, dim=1, keepdim=True)  # (batch, 1, 2)
            centroid_target = torch.mean(target_corr, dim=1, keepdim=True)          # (batch, 1, 2)
            
            # Center the point clouds.
            source_centered = source_transformed - centroid_source  # (batch, N, 2)
            target_centered = target_corr - centroid_target         # (batch, N, 2)
            
            # Compute cross-covariance matrix H for each batch element.
            H = torch.bmm(source_centered.transpose(1, 2), target_centered)  # (batch, 2, 2)
            
            # Perform SVD on H.
            U, S, Vh = torch.linalg.svd(H, full_matrices=False)
            V = Vh.transpose(-2, -1)
            R_delta = torch.bmm(V, U.transpose(1, 2))  # (batch, 2, 2)
            
            # Compute translation delta.
            t_delta = centroid_target.transpose(1, 2) - torch.bmm(R_delta, centroid_source.transpose(1, 2))  # (batch, 2, 1)
            
            # Convert delta transformation to vector form: [delta_theta, delta_tx, delta_ty]
            delta_theta = torch.atan2(R_delta[:, 1, 0], R_delta[:, 0, 0])  # (batch,)
            delta_tx = t_delta[:, 0, 0]  # (batch,)
            delta_ty = t_delta[:, 1, 0]  # (batch,)
            delta_vec = torch.stack([delta_theta, delta_tx, delta_ty], dim=1)  # (batch, 3)
            
            # Convert delta transformation to homogeneous matrix.
            delta_T = vec2mat(delta_vec)  # (batch, 3, 3)
            
            # Update cumulative transformation: new T = delta_T * T.
            T = torch.bmm(delta_T, T)
            
            # Update transformed source points.
            source_transformed_h = torch.bmm(source_h, T.transpose(1, 2))
            source_transformed = source_transformed_h[:, :, :2]
        
        # Extract refined transformation from T.
        refined_theta = torch.atan2(T[:, 1, 0], T[:, 0, 0])
        refined_tx = T[:, 0, 2]
        refined_ty = T[:, 1, 2]
        refined_transformation = torch.stack([refined_theta, refined_tx, refined_ty], dim=1)
        return refined_transformation

class FeatureExtractor(nn.Module):
    """
    Extracts features from a 1D lidar scan using Conv1D layers.
    Input shape: (batch_size, 1, 360)
    Output shape: (batch_size, feature_dim)
    """
    def __init__(self, in_channels=1, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, feature_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)         # Shape: (batch_size, feature_dim, 1)
        x = x.squeeze(-1)        # Shape: (batch_size, feature_dim)
        return x

class InitialTransformationEstimator(nn.Module):
    """
    Uses a simple MLP to map extracted features to an initial 2D pose estimate.
    For 2D, the transformation is represented as [theta, tx, ty],
    where theta is the rotation angle.
    """
    def __init__(self, input_dim=64, output_dim=3):
        super(InitialTransformationEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, features):
        x = nn.functional.relu(self.fc1(features))
        trans = self.fc2(x)
        return trans

class LocalizationNet(nn.Module):
    """
    Combines feature extraction, initial transformation estimation,
    and differentiable ICP into an end-to-end trainable localization network.
    """
    def __init__(self):
        super(LocalizationNet, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels=1, feature_dim=64)
        self.init_trans_estimator = InitialTransformationEstimator(input_dim=64, output_dim=3)
        self.diff_icp = DifferentiableICP(iterations=5)
    
    def forward(self, lidar_scan, source_points, target_points):
        # Ensure inputs are float32
        lidar_scan = lidar_scan.float()
        source_points = source_points.float()
        target_points = target_points.float()
        # Reshape lidar_scan to (batch, 1, 360)
        x = lidar_scan.unsqueeze(1)
        features = self.feature_extractor(x)
        # Get initial transformation estimate from lidar scan features.
        init_trans = self.init_trans_estimator(features)
        # Refine transformation using Differentiable ICP.
        refined_trans = self.diff_icp(source_points, target_points, init_trans)
        return refined_trans

# ----------------------------- Testing and Visualization -----------------------------

def generate_synthetic_data(num_points=360, radius=5.0, noise_std=1.0):
    """
    Generates a synthetic lidar scan and corresponding 2D point cloud.
    Returns:
      lidar_scan: 1D numpy array of length num_points (range measurements)
      source_points: 2D numpy array of shape (num_points, 2)
    """
    angles = np.linspace(np.pi, -np.pi, num_points, endpoint=False)
    # Use a fixed radius plus optional noise.
    ranges = radius * np.ones(num_points) + np.random.randn(num_points) * noise_std
    source_points = np.stack([ranges * np.cos(angles), ranges * np.sin(angles)], axis=1)
    lidar_scan = ranges.astype(np.float32)
    return lidar_scan, source_points

def apply_transformation(points, trans):
    """
    Applies a 2D transformation (given as [theta, tx, ty]) to a point cloud.
    points: numpy array of shape (N, 2)
    trans: list or array [theta, tx, ty]
    Returns:
      transformed_points: numpy array of shape (N, 2)
    """
    batch_points = torch.from_numpy(points).float().unsqueeze(0)  # (1, N, 2)
    T = vec2mat(torch.tensor(trans, dtype=torch.float32).unsqueeze(0))  # (1, 3, 3)
    N = batch_points.shape[1]
    ones = torch.ones((1, N, 1))
    points_h = torch.cat([batch_points, ones], dim=2)  # (1, N, 3)
    transformed_h = torch.bmm(points_h, T.transpose(1,2))
    transformed = transformed_h[:, :, :2].squeeze(0).detach().numpy()
    return transformed

def test_localization_net():
    # Define ground truth transformation: [theta, tx, ty]
    gt_transformation = [0.2, 0.5, -0.3]
    
    # Generate synthetic source data.
    lidar_scan, source_points = generate_synthetic_data(num_points=360, radius=1.0, noise_std=0.01)
    
    # Generate target points by applying the ground truth transformation.
    target_points = apply_transformation(source_points, gt_transformation)
    
    # Convert data to tensors.
    lidar_scan_tensor = torch.tensor(lidar_scan).unsqueeze(0)          # (1, 360)
    source_points_tensor = torch.tensor(source_points).unsqueeze(0)      # (1, 360, 2)
    target_points_tensor = torch.tensor(target_points).unsqueeze(0)      # (1, 360, 2)
    
    # Instantiate the LocalizationNet model.
    model = LocalizationNet()
    
    # For testing the ICP module in isolation, we override the initial transformation.
    # Here we simulate an initial guess that's slightly off from the ground truth.
    with torch.no_grad():
        fixed_initial_trans = torch.tensor([[0.25, 0.45, -0.25]], dtype=torch.float32)  # (1, 3)
        refined_transformation = model.diff_icp(source_points_tensor, target_points_tensor, fixed_initial_trans)
    
    print("Ground Truth Transformation:", gt_transformation)
    print("Fixed Initial Transformation:", fixed_initial_trans.numpy()[0])
    print("Refined Transformation:", refined_transformation.detach().numpy()[0])
    
    # Visualization:
    # 1. Plot the source point cloud, target point cloud, and the source transformed by the refined transformation.
    refined_T = vec2mat(refined_transformation)  # (1, 3, 3)
    ones = torch.ones((1, source_points_tensor.shape[1], 1))
    source_h = torch.cat([source_points_tensor, ones], dim=2)
    source_h = source_h.float()
    transformed_source_h = torch.bmm(source_h, refined_T.transpose(1,2))
    transformed_source = transformed_source_h[0,:,:2].detach().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(source_points[:, 0], source_points[:, 1], label="Source", c="blue", alpha=0.5)
    plt.scatter(target_points[:, 0], target_points[:, 1], label="Target", c="green", alpha=0.5)
    plt.scatter(transformed_source[:, 0], transformed_source[:, 1], label="Transformed Source", c="red", marker="x")
    plt.legend()
    plt.title("Differentiable ICP Refinement")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_localization_net()
