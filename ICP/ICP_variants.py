import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import logging

def point_to_point_icp(source, target, max_iterations=50, threshold=1e-6):
    """
    Point-to-Point ICP algorithm for 2D and 3D point clouds.
    Matches each source point to the nearest target point and aligns based on their centroids.
    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.
    
    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    dim = source.shape[1]  # Dimensionality (2D or 3D)
    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    for _ in range(max_iterations):
        # Compute distances and find nearest neighbors
        distances = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
        correspondences = np.argmin(distances, axis=1)

        # Compute centroids
        source_centroid = source.mean(axis=0)
        target_centroid = target[correspondences].mean(axis=0)

        # Demean the points
        source_demean = source - source_centroid
        target_demean = target[correspondences] - target_centroid

        # Compute the cross-covariance matrix
        W = source_demean.T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = target_centroid - R @ source_centroid

        # Update transformation
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R  # Rotation matrix
        current_transformation[:dim, dim] = t  # Translation vector
        transformation = current_transformation @ transformation

        # Transform source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences))

        # Check for convergence
        mean_error = np.mean(np.linalg.norm(source - target[correspondences], axis=1))
        if mean_error < threshold:
            break

    return transformation

def point_to_plane_icp(source, target, normals, max_iterations=50, threshold=1e-6):
    """
    Point-to-Plane ICP algorithm for 2D and 3D point clouds.
    Minimizes the distance between a source point and the plane defined by a target point and its normal.
    Converges faster than Point-to-Point ICP when point clouds are smooth or continuous.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        normals (ndarray): Surface normals of the target point cloud (Nx2 or Nx3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert source.shape == target.shape, "Source and target must have the same shape."
    #assert source.shape == normals.shape, "Normals must correspond to target points."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."

    # Initialize transformation matrix and steps list
    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Tree for efficient nearest neighbor search
    tree = KDTree(target)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, correspondences = tree.query(source)
        
        # Check for empty correspondences
        if len(correspondences) == 0:
            logger.warning("No correspondences found. Terminating ICP.")
            break

        # Corresponding target normals
        target_normals = normals[correspondences]

        # Compute error vector: distance along normal direction
        diff = target[correspondences] - source
        errors = np.sum(diff * target_normals, axis=1)  # Project differences onto normals

        # Assemble the Jacobian matrix (A) and residual vector (b)
        if dim == 2:
            A = np.zeros((len(source), 3))  # [tx, ty, rotation]
            A[:, :2] = target_normals
            A[:, 2] = source[:, 0] * target_normals[:, 1] - source[:, 1] * target_normals[:, 0]
        elif dim == 3:
            A = np.zeros((len(source), 6))  # [tx, ty, tz, rx, ry, rz]
            A[:, :3] = target_normals
            A[:, 3:] = np.cross(source, target_normals)

        b = errors

        # Solve for the transformation parameters using weighted least squares
        # Optional: Incorporate weights to reduce the influence of outliers
        weights = 1.0 / (distances + 1e-8)  # Add epsilon to prevent division by zero
        A_weighted = A * weights[:, np.newaxis]
        b_weighted = b * weights
        x, residuals, rank, s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)

        # Construct transformation matrix
        current_transformation = np.eye(dim + 1)
        if dim == 2:
            theta = x[2] % (2 * np.pi)  # Normalize angle
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            t = x[:2]
        elif dim == 3:
            theta = np.linalg.norm(x[3:])
            if theta > 1e-12:
                k = x[3:] / theta
                K = np.array([
                    [0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]
                ])
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            else:
                R = np.eye(3)
            t = x[:3]

        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t
        transformation = current_transformation @ transformation

        # Transform source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences))

        # Compute mean error and check for convergence
        mean_error = np.mean(np.abs(errors))
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info("Convergence achieved.")
            break
        previous_error = mean_error

    return transformation

def trimmed_icp(source, target, trim_ratio=0.7, max_iterations=100, threshold=1e-6):
    """
    Trimmed ICP algorithm for 2D and 3D point clouds.
    Minimizes the distance between a source point and the target point.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        trim_ratio (float): Ratio of closest points to consider (0 < trim_ratio <= 1).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert 0 < trim_ratio <= 1, "trim_ratio must be within (0, 1]."
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] > 0, "Point clouds must not be empty."

    dim = source.shape[0]  # Dimensionality (2D or 3D)
    #assert dim in [2, 3], "Only 2D and 3D point clouds are supported."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Tree for efficient nearest neighbor search
    tree = KDTree(target)

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, correspondences = tree.query(source)

        # Trim correspondences based on trim_ratio
        num_trim = int(np.ceil(trim_ratio * len(source)))
        sorted_indices = np.argsort(distances)
        trimmed_indices = sorted_indices[:num_trim]

        trimmed_source = source[trimmed_indices]
        trimmed_correspondences = correspondences[trimmed_indices]
        trimmed_target = target[trimmed_correspondences]

        # Check if sufficient correspondences exist
        if len(trimmed_source) < dim + 1:
            logger.warning(f"Iteration {iteration + 1}: Not enough correspondences to compute transformation.")
            break

        # Compute centroids
        source_centroid = trimmed_source.mean(axis=0)
        target_centroid = trimmed_target.mean(axis=0)

        # Demean the points
        source_demean = trimmed_source - source_centroid
        target_demean = trimmed_target - target_centroid

        # Compute the cross-covariance matrix
        W = source_demean.T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure a proper rotation matrix (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), trimmed_correspondences.copy()))

        # Compute mean error for convergence
        mean_error = np.mean(distances[trimmed_indices])
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if mean_error < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break

    return transformation

def generalized_icp(source, target, source_cov, target_cov, max_iterations=50, threshold=1e-6):
    """
    Generalized ICP algorithm for 2D and 3D point clouds.
    Incorporates point-wise covariance matrices to account for uncertainty.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        source_cov (ndarray): Covariance matrices of source points (Nx2x2 or Nx3x3).
        target_cov (ndarray): Covariance matrices of target points (Nx2x2 or Nx3x3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] == source_cov.shape[0] == target_cov.shape[0], "Covariance matrices must correspond to point clouds."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."
    assert source_cov.shape[1:] == target_cov.shape[1:] == (dim, dim), "Covariance matrices must be of shape (N, dim, dim)."
    assert max_iterations > 0, "max_iterations must be positive."
    assert threshold > 0, "threshold must be positive."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Tree for efficient nearest neighbor search
    tree = KDTree(target)

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, correspondences = tree.query(source)

        # Extract corresponding target points and covariances
        target_corr = target[correspondences]
        target_cov_corr = target_cov[correspondences]

        # Compute combined covariances (assume source_cov is diagonal or symmetric)
        # Add a small regularization term to avoid singularity
        combined_cov = source_cov + target_cov_corr + np.eye(dim) * 1e-8

        # Invert combined covariance matrices
        try:
            inv_combined_cov = np.linalg.inv(combined_cov)
        except np.linalg.LinAlgError:
            logger.warning(f"Iteration {iteration + 1}: Combined covariance matrix is singular. Adding regularization.")
            combined_cov += np.eye(dim) * 1e-8
            inv_combined_cov = np.linalg.inv(combined_cov)

        # Compute weighted differences
        diff = target_corr - source  # (N, dim)
        weighted_diff = np.einsum('nij,nj->ni', inv_combined_cov, diff)  # (N, dim)

        # Compute centroids
        source_centroid = source.mean(axis=0)
        target_centroid = target_corr.mean(axis=0)

        # Demean the points
        source_demean = source - source_centroid
        target_demean = target_corr - target_centroid

        # Compute cross-covariance matrix
        W = source_demean.T @ weighted_diff  # (dim, dim)

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure a proper rotation matrix (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences.copy()))

        # Compute mean error for convergence
        mean_error = np.mean(np.linalg.norm(diff, axis=1))
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if mean_error < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break

    return transformation, steps

def robust_icp(source, target, max_iterations=50, threshold=1e-6, loss_function="huber", loss_param=1.0):
    """
    Robust ICP algorithm for 2D and 3D point clouds.
    Minimizes the distance between a source point and the target point using robust loss functions.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.
        loss_function (str): Type of robust loss ("huber" or "tukey").
        loss_param (float): Parameter for the loss function (e.g., delta for Huber).

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    #assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] > 0, "Point clouds must not be empty."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."
    assert 0 < loss_param <= 1, "trim_ratio must be within (0, 1]."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Tree for efficient nearest neighbor search
    tree = KDTree(target)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, correspondences = tree.query(source)

        # Compute residuals
        residuals = target[correspondences] - source

        # Apply robust loss
        norm_residuals = np.linalg.norm(residuals, axis=1)
        if loss_function == "huber":
            # Huber loss
            weights = np.where(
                norm_residuals <= loss_param,
                1.0,
                loss_param / norm_residuals
            )
        elif loss_function == "tukey":
            # Tukey’s biweight function
            weights = np.where(
                norm_residuals <= loss_param,
                (1 - (norm_residuals / loss_param) ** 2) ** 2,
                0
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # Avoid division by zero
        weight_sum = weights.sum()
        if weight_sum == 0:
            logger.warning(f"Iteration {iteration + 1}: All weights are zero. Terminating ICP.")
            break

        # Weight the residuals
        weighted_residuals = residuals * weights[:, None]

        # Compute centroids
        source_centroid = (source * weights[:, None]).sum(axis=0) / weight_sum
        target_centroid = (target[correspondences] * weights[:, None]).sum(axis=0) / weight_sum

        # Demean the points
        source_demean = source - source_centroid
        target_demean = target[correspondences] - target_centroid

        # Compute cross-covariance matrix
        W = (source_demean * weights[:, None]).T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences.copy()))

        # Compute weighted mean error for convergence
        mean_error = np.mean(weights * norm_residuals)
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break
        previous_error = mean_error

    return transformation

def colored_icp(source, target, source_colors, target_colors, max_iterations=50, threshold=1e-6, color_weight=0.5):
    """
    Colored ICP algorithm for 2D and 3D point clouds.
    Incorporates color information into the ICP alignment process.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        source_colors (ndarray): Colors associated with source points (Nx3, RGB).
        target_colors (ndarray): Colors associated with target points (Nx3, RGB).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.
        color_weight (float): Weight for color error in the combined loss (0 to 1).

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] == source_colors.shape[0] == target_colors.shape[0], "Color arrays must correspond to point clouds."
    assert source.shape[1] in [2, 3], "Only 2D and 3D point clouds are supported."
    dim = source.shape[1]
    assert color_weight >= 0 and color_weight <= 1, "color_weight must be between 0 and 1."
    assert max_iterations > 0, "max_iterations must be positive."
    assert threshold > 0, "threshold must be positive."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Tree for efficient nearest neighbor search
    tree = KDTree(target)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors based on geometry
        distances, correspondences = tree.query(source)

        # Compute color differences
        color_diff = target_colors[correspondences] - source_colors
        color_distances = np.linalg.norm(color_diff, axis=1)

        # Normalize geometric and color distances to balance their contributions
        geom_distances = np.linalg.norm(source - target[correspondences], axis=1)
        # Prevent division by zero by adding a small epsilon
        geom_distances_normalized = geom_distances / (geom_distances.max() + 1e-8)
        color_distances_normalized = color_distances / (color_distances.max() + 1e-8)

        # Combine geometric and color-based distances
        combined_distances = (1 - color_weight) * geom_distances_normalized + \
                             color_weight * color_distances_normalized

        # Apply weighting based on combined distances using inverse weighting
        weights = 1 / (combined_distances + 1e-8)  # Prevent division by zero

        # Avoid division by zero
        weight_sum = weights.sum()
        if weight_sum == 0:
            logger.warning(f"Iteration {iteration + 1}: All weights are zero. Terminating ICP.")
            break

        # Compute weighted centroids
        source_centroid = (source * weights[:, None]).sum(axis=0) / weight_sum
        target_centroid = (target[correspondences] * weights[:, None]).sum(axis=0) / weight_sum

        # Demean the points
        source_demean = source - source_centroid
        target_demean = target[correspondences] - target_centroid

        # Compute cross-covariance matrix
        W = (source_demean * weights[:, None]).T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences.copy()))

        # Compute mean error for convergence
        mean_error = np.mean(combined_distances)
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break
        previous_error = mean_error

    return transformation, steps

def sparse_icp(source, target, sparsity=0.1, max_iterations=50, threshold=1e-6):
    """
    Sparse ICP algorithm for 2D and 3D point clouds.
    
    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        sparsity (float): Fraction of points to retain (0 < sparsity <= 1).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.
    
    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert 0 < sparsity <= 1, "sparsity must be within (0, 1]."
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] > 0, "Point clouds must not be empty."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Downsample source and target point clouds
    num_source_points = max(int(len(source) * sparsity), dim + 1)
    num_target_points = max(int(len(target) * sparsity), dim + 1)

    source_indices = np.random.choice(len(source), num_source_points, replace=False)
    target_indices = np.random.choice(len(target), num_target_points, replace=False)

    source_sparse = source[source_indices]
    target_sparse = target[target_indices]

    # Initialize KD-Tree for efficient nearest neighbor search
    tree = KDTree(target_sparse)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, correspondences = tree.query(source_sparse)
        correspondences = correspondences.flatten()

        # Check if sufficient correspondences exist
        if len(correspondences) < dim + 1:
            logger.warning(f"Iteration {iteration + 1}: Not enough correspondences to compute transformation.")
            break

        # Compute centroids
        source_centroid = source_sparse.mean(axis=0)
        target_centroid = target_sparse[correspondences].mean(axis=0)

        # Demean the points
        source_demean = source_sparse - source_centroid
        target_demean = target_sparse[correspondences] - target_centroid

        # Compute cross-covariance matrix
        W = source_demean.T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source_sparse = (R @ source_sparse.T).T + t

        # Save intermediate steps for visualization
        steps.append((source_sparse.copy(), correspondences.copy()))

        # Compute mean error for convergence
        mean_error = np.mean(distances)
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break
        previous_error = mean_error

    return transformation


def multi_scale_icp(source, target, scales=[0.2, 0.5, 1.0], max_iterations=50, threshold=1e-6):
    """
    Multi-Scale ICP algorithm for 2D and 3D point clouds.
    
    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        scales (list): List of scales for multi-resolution processing (0 < scale <= 1).
        max_iterations (int): Maximum number of ICP iterations per scale.
        threshold (float): Convergence threshold for mean error.
    
    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        all_steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    #assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] > 0, "Point clouds must not be empty."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."
    for scale in scales:
        assert 0 < scale <= 1, "All scales must be within (0, 1]."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    all_steps = []

    for scale in scales:
        # Downsample source and target point clouds
        num_source_points = max(int(len(source) * scale), dim + 1)
        num_target_points = max(int(len(target) * scale), dim + 1)

        source_indices = np.random.choice(len(source), num_source_points, replace=False)
        target_indices = np.random.choice(len(target), num_target_points, replace=False)

        source_downsampled = source[source_indices]
        target_downsampled = target[target_indices]

        # Initialize KD-Tree for efficient nearest neighbor search
        tree = KDTree(target_downsampled)

        previous_error = float('inf')

        for iteration in range(max_iterations):
            # Find nearest neighbors
            distances, correspondences = tree.query(source_downsampled)
            correspondences = correspondences.flatten()

            # Check if sufficient correspondences exist
            if len(correspondences) < dim + 1:
                logger.warning(f"Scale {scale}, Iteration {iteration + 1}: Not enough correspondences to compute transformation.")
                break

            # Compute centroids
            source_centroid = source_downsampled.mean(axis=0)
            target_centroid = target_downsampled[correspondences].mean(axis=0)

            # Demean the points
            source_demean = source_downsampled - source_centroid
            target_demean = target_downsampled[correspondences] - target_centroid

            # Compute cross-covariance matrix
            W = source_demean.T @ target_demean

            # Singular Value Decomposition (SVD)
            U, _, Vt = np.linalg.svd(W)
            R = Vt.T @ U.T

            # Ensure proper rotation matrix
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Compute translation
            t = target_centroid - R @ source_centroid

            # Construct the current transformation matrix
            current_transformation = np.eye(dim + 1)
            current_transformation[:dim, :dim] = R
            current_transformation[:dim, dim] = t

            # Update the overall transformation
            transformation = current_transformation @ transformation

            # Transform the source points
            source_downsampled = (R @ source_downsampled.T).T + t
            source = (R @ source.T).T + t  # Apply transformation to full source for next scale

            # Save intermediate steps for visualization
            all_steps.append((source_downsampled.copy(), correspondences.copy()))

            # Compute mean error for convergence
            mean_error = np.mean(distances)
            logger.info(f"Scale {scale}, Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

            if abs(previous_error - mean_error) < threshold:
                logger.info(f"Scale {scale}, Convergence achieved at iteration {iteration + 1}.")
                break
            previous_error = mean_error

    return transformation


def symmetric_icp(source, target, max_iterations=50, threshold=1e-6):
    """
    Symmetric ICP algorithm for 2D and 3D point clouds.
    
    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.
    
    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source.shape[0] > 0, "Point clouds must not be empty."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Trees for efficient nearest neighbor search
    tree_source = KDTree(source)
    tree_target = KDTree(target)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors (source-to-target)
        distances_st, correspondences_st = tree_target.query(source)
        correspondences_st = correspondences_st.flatten()

        # Find nearest neighbors (target-to-source)
        distances_ts, correspondences_ts = tree_source.query(target)
        correspondences_ts = correspondences_ts.flatten()

        # Symmetric correspondences
        mutual = np.where(correspondences_ts[correspondences_st] == np.arange(len(source)))[0]
        if len(mutual) == 0:
            logger.warning(f"Iteration {iteration + 1}: No mutual correspondences found.")
            break

        source_corr = source[mutual]
        target_corr = target[correspondences_st[mutual]]

        # Compute centroids
        source_centroid = source_corr.mean(axis=0)
        target_centroid = target_corr.mean(axis=0)

        # Demean the points
        source_demean = source_corr - source_centroid
        target_demean = target_corr - target_centroid

        # Compute cross-covariance matrix
        W = source_demean.T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences_st[mutual].copy()))

        # Compute mean error for convergence
        mean_error = np.mean(distances_st[mutual])
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break
        previous_error = mean_error

    return transformation

def normal_space_icp(source, target, source_normals, target_normals, max_iterations=50, threshold=1e-6):
    """
    Normal-Space ICP algorithm for 2D and 3D point clouds.
    
    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        source_normals (ndarray): Normals of source points (Nx2 or Nx3).
        target_normals (ndarray): Normals of target points (Nx2 or Nx3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.
    
    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source_normals.shape == target_normals.shape, "Source and target normals must have the same shape."
    assert source.shape[0] > 0, "Point clouds must not be empty."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # Initialize KD-Trees for efficient nearest neighbor search
    tree_target = KDTree(target)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors based on position and normal similarity
        distances = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
        normal_differences = np.linalg.norm(source_normals[:, None, :] - target_normals[None, :, :], axis=2)
        combined_distances = distances + 0.1 * normal_differences
        correspondences = np.argmin(combined_distances, axis=1)

        # Extract correspondences
        source_corr = source
        target_corr = target[correspondences]

        # Compute centroids
        source_centroid = source_corr.mean(axis=0)
        target_centroid = target_corr.mean(axis=0)

        # Demean the points
        source_demean = source_corr - source_centroid
        target_demean = target_corr - target_centroid

        # Compute cross-covariance matrix
        W = source_demean.T @ target_demean

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the current transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Transform the source points
        source = (R @ source.T).T + t

        # Save intermediate steps for visualization
        steps.append((source.copy(), correspondences.copy()))

        # Compute mean error for convergence
        mean_error = np.mean(combined_distances[np.arange(len(source)), correspondences])
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break
        previous_error = mean_error

    return transformation, steps


def probabilistic_icp(source, target, source_cov, target_cov, max_iterations=50, threshold=1e-6):
    """
    Probabilistic ICP algorithm for 2D and 3D point clouds.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        source_cov (ndarray): Covariance matrices of source points (Nx2x2 or Nx3x3).
        target_cov (ndarray): Covariance matrices of target points (Nx2x2 or Nx3x3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error.

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences for animation.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input validation
    assert source.shape == target.shape, "Source and target must have the same shape."
    assert source_cov.shape == target_cov.shape, "Source and target covariance matrices must have the same shape."
    assert source.shape[0] == source_cov.shape[0] == target_cov.shape[0], "Covariance matrices must correspond to point clouds."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."
    assert source_cov.shape[1:] == target_cov.shape[1:] == (dim, dim), "Covariance matrices must be of shape (N, dim, dim)."
    assert max_iterations > 0, "max_iterations must be positive."
    assert threshold > 0, "threshold must be positive."

    transformation = np.eye(dim + 1)
    steps = []
    tree = KDTree(target)

    previous_error = float('inf')

    for iteration in range(max_iterations):
        distances, correspondences = tree.query(source)
        correspondences = correspondences.flatten()

        target_corr = target[correspondences]
        target_cov_corr = target_cov[correspondences]

        # Construct combined covariance: NxDxD
        combined_cov = source_cov + target_cov_corr + np.eye(dim)[None, :, :] * 1e-8

        # Invert each covariance matrix individually
        inv_combined_cov = []
        for c in combined_cov:
            try:
                inv_c = np.linalg.inv(c)
            except np.linalg.LinAlgError:
                # If singular, add more regularization
                c_reg = c + np.eye(dim) * 1e-8
                inv_c = np.linalg.inv(c_reg)
            inv_combined_cov.append(inv_c)
        inv_combined_cov = np.array(inv_combined_cov)

        diff = target_corr - source  # (N, dim)
        # weighted_diff: (N, dim)
        weighted_diff = np.einsum('nij,nj->ni', inv_combined_cov, diff)

        source_centroid = source.mean(axis=0)
        target_centroid = target_corr.mean(axis=0)

        source_demean = source - source_centroid
        target_demean = target_corr - target_centroid

        W = source_demean.T @ weighted_diff
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = target_centroid - R @ source_centroid

        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t
        transformation = current_transformation @ transformation

        source = (R @ source.T).T + t

        steps.append((source.copy(), correspondences.copy()))

        mean_error = np.mean(np.linalg.norm(diff, axis=1))
        logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        if abs(previous_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break
        previous_error = mean_error

    return transformation, steps

def coarse_to_fine_icp(source, target, levels=[4, 2, 1], max_iterations=50, threshold=1e-6):
    """
    Coarse-to-Fine ICP implementation.
    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Nx2 or Nx3).
        levels (list): List of downsampling factors (e.g., [4, 2, 1]).
        max_iterations (int): Maximum number of iterations at each level.
        threshold (float): Convergence threshold for mean error.
    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
    """
    dim = source.shape[1]
    transformation = np.eye(dim + 1)  # Initial transformation

    for level in levels:
        # Downsample source and target point clouds
        source_ds = downsample_point_cloud(source, level)
        target_ds = downsample_point_cloud(target, level)

        # Apply ICP at the current resolution
        current_transformation = multi_scale_icp(source_ds, target_ds, max_iterations, threshold)

        # Update the source points with the current transformation
        R_mat = current_transformation[:dim, :dim]
        t = current_transformation[:dim, dim]
        source = (R_mat @ source.T).T + t

        # Update the overall transformation
        transformation = current_transformation @ transformation

    return transformation

def downsample_point_cloud(points, factor):
    """
    Downsample the point cloud by selecting every nth point.
    Args:
        points (ndarray): Original point cloud (Nx2 or Nx3).
        factor (int): Downsampling factor.
    Returns:
        ndarray: Downsampled point cloud.
    """
    return points[::factor]