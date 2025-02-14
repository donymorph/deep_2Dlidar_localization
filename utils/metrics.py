import numpy as np
from scipy.spatial.distance import directed_hausdorff

#$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (e_i - \bar{e})^2}$$
def compute_rmse(gt_array, pred_array):
    """Computes Root Mean Square Error (RMSE)."""
    if len(gt_array) == len(pred_array) and len(gt_array) > 0:
        squared_errors = np.square(gt_array - pred_array)
        rmse = np.sqrt(np.mean(squared_errors))
        return rmse
    return 0.0

def compute_mae_L1(gt_array, pred_array):
    """Computes Mean Absolute Error (MAE)."""
    if len(gt_array) == len(pred_array) and len(gt_array) > 0:
        mae = np.mean(np.abs(gt_array - pred_array))
        return mae
    return 0.0

def compute_std_error(gt_array, pred_array):
    """Computes standard deviation of error."""
    if len(gt_array) == len(pred_array) and len(gt_array) > 0:
        errors = np.linalg.norm(gt_array - pred_array, axis=1)
        return np.std(errors)
    return 0.0

#$$\text{MedAE} = \text{median}(|x_i - \hat{x}_i|)$$
def compute_medae(gt_array, pred_array):
    """Computes Median Absolute Error (MedAE), which is robust to outliers."""
    if len(gt_array) == len(pred_array) and len(gt_array) > 0:
        errors = np.linalg.norm(gt_array - pred_array, axis=1)
        return np.median(errors)
    return 0.0

def compute_relative_error(gt_array, pred_array):
    """Computes Relative Error (%) per sample."""
    if len(gt_array) == len(pred_array) and len(gt_array) > 0:
        relative_errors = np.abs(gt_array - pred_array) / (np.abs(gt_array) + 1e-6)  # Avoid division by zero
        return np.mean(relative_errors) * 100  # Convert to percentage
    return 0.0


def compute_hausdorff_distance(gt_array, pred_array):
    """Computes the Hausdorff Distance between ground truth and predicted trajectories."""
    if len(gt_array) > 0 and len(pred_array) > 0:
        d1 = directed_hausdorff(gt_array, pred_array)[0]
        d2 = directed_hausdorff(pred_array, gt_array)[0]
        return max(d1, d2)
    return 0.0
