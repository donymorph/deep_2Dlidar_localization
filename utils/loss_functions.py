import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    """
    Improved loss function for predicting position (x, y) and orientation (z).

    - For position: uses Smooth L1 Loss, which is less sensitive to outliers than MSE.
    - For orientation: uses a cosine-based loss, i.e. 1 - cos(delta_angle),
      which avoids discontinuities due to angle wrapping.
    - The final loss is a weighted sum of the two.
    """
    def __init__(self, lambda_pos=1.0, lambda_orient=1.0):
        """
        Args:
            lambda_pos (float): Weight for the position loss.
            lambda_orient (float): Weight for the orientation loss.
        """
        super(PoseLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_orient = lambda_orient

    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of shape (batch_size, 3) -> [x_pred, y_pred, z_pred]
            target: Tensor of shape (batch_size, 3) -> [x_true, y_true, z_true]
        
        Returns:
            Combined weighted loss.
        """
        # Extract position and orientation components
        pos_pred = pred[:, :2]     # (x, y)
        pos_target = target[:, :2]
        orient_pred = pred[:, 2]     # orientation z (angle in radians)
        orient_target = target[:, 2]
        
        # Position loss: Smooth L1 Loss (robust regression loss)
        pos_loss = F.huber_loss(pos_pred, pos_target, reduction='mean')
        
        # Orientation loss: 1 - cos(angle difference)
        # This formulation makes sure that differences like 359° and 1° are considered small.
        orient_loss = torch.mean(1 - torch.cos(orient_pred - orient_target))
        
        # Combine the losses using weighted sum
        total_loss = self.lambda_pos * pos_loss + self.lambda_orient * orient_loss
        
        return total_loss

class LogCoshPoseLoss(nn.Module):
    """
    LogCoshPoseLoss computes the loss for predicting position (x, y) and orientation (z)
    using the log-cosh loss for both components. It handles angle wrapping for orientation.
    
    Args:
        lambda_pos (float): Weight for the position loss.
        lambda_orient (float): Weight for the orientation loss.
    """
    def __init__(self, lambda_pos=1.0, lambda_orient=1.0):
        super(LogCoshPoseLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_orient = lambda_orient

    def log_cosh(self, x):
        """
        Numerically stable implementation of log-cosh.
        """
        return torch.log(torch.cosh(x + 1e-12))  # small constant for numerical stability

    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of shape (batch_size, 3) with predictions [x, y, z] (z in radians)
            target: Tensor of shape (batch_size, 3) with ground truth values [x, y, z]
            
        Returns:
            Combined weighted loss.
        """
        # Split position and orientation components
        pos_pred = pred[:, :2]      # x, y
        pos_target = target[:, :2]
        orient_pred = pred[:, 2]      # orientation z (angle in radians)
        orient_target = target[:, 2]
        
        # Position difference loss using log-cosh loss
        pos_diff = pos_pred - pos_target
        pos_loss = self.log_cosh(pos_diff).mean()  # average over batch and dimensions

        # Orientation loss:
        # First, compute the angle difference and wrap it to [-pi, pi]
        angle_diff = orient_pred - orient_target
        angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
        orient_loss = self.log_cosh(angle_diff).mean()
        
        # Combine losses with weights
        total_loss = self.lambda_pos * pos_loss + self.lambda_orient * orient_loss
        return total_loss
# Example usage:
if __name__ == "__main__":
    loss_fn = PoseLoss(lambda_pos=1.0, lambda_orient=0.5)

    # Example predictions and ground truth (angles in radians)
    predictions = torch.tensor([[1.2, 2.3, 0.5],
                                [4.1, -3.2, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 2.5, 0.6],
                            [4.0, -3.0, 1.2]], dtype=torch.float32)

    loss = loss_fn(predictions, targets)
    print(f"Improved Pose Loss: {loss.item():.4f}")
