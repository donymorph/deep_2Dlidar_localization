import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
import logging

def point_to_point_icp(source, target, max_iterations=50, threshold=1e-6, 
                       initial_transform=None, outlier_rejection_threshold=0.5):
    """
    Improved Point-to-Point ICP algorithm for 2D/3D point clouds.
    
    Enhancements:
    - Convergence check (stops early when mean error stabilizes).
    - Outlier rejection (ignores high-distance correspondences).
    - Weighted correspondences (based on confidence from distances).
    - Supports an initial transformation.

    Args:
        source (ndarray): Source point cloud of shape (N, 2) or (N, 3).
        target (ndarray): Target point cloud of shape (M, 2) or (M, 3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for mean error change.
        initial_transform (ndarray, optional): Initial transformation matrix (3x3 for 2D, 4x4 for 3D).
        outlier_rejection_threshold (float, optional): Maximum distance to consider a match valid.

    Returns:
        transformation (ndarray): Final transformation (3x3 for 2D, 4x4 for 3D).
        steps (list): List of transformation matrices for debugging.
    """
    dim = source.shape[1]  # Determine 2D or 3D
    transformation = np.eye(dim + 1) if initial_transform is None else initial_transform
    steps = []
    
    # Apply initial transformation if provided
    if initial_transform is not None:
        source = (initial_transform[:dim, :dim] @ source.T).T + initial_transform[:dim, dim]

    # KD-tree for nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)

    prev_error = float('inf')  # Track previous iteration error

    for iteration in range(max_iterations):
        # Step 1: Find the nearest neighbors in the target
        distances, indices = nbrs.kneighbors(source)
        distances = distances.ravel()
        correspondences = indices.ravel()
        
        # Step 2: Outlier rejection (ignore pairs with large distances)
        if outlier_rejection_threshold:
            valid_mask = distances < outlier_rejection_threshold
            source_filtered = source[valid_mask]
            target_filtered = target[correspondences][valid_mask]
        else:
            source_filtered = source
            target_filtered = target[correspondences]

        # Ensure we have enough valid matches
        if len(source_filtered) < 3:
            print("Not enough valid correspondences, stopping ICP.")
            break

        # Step 3: Compute centroids of the matched points
        source_centroid = source_filtered.mean(axis=0)
        target_centroid = target_filtered.mean(axis=0)

        # Step 4: Demean both point sets
        source_demean = source_filtered - source_centroid
        target_demean = target_filtered - target_centroid

        # Step 5: Compute cross-covariance matrix
        W = source_demean.T @ target_demean

        # Step 6: Compute the best rotation using SVD
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure a valid rotation (fix reflections if necessary)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Step 7: Compute translation
        t = target_centroid - R @ source_centroid

        # Step 8: Construct transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update the overall transformation
        transformation = current_transformation @ transformation

        # Step 9: Apply transformation to source
        source = (R @ source.T).T + t

        # Step 10: Compute mean error and check convergence
        mean_error = np.mean(distances[valid_mask]) if outlier_rejection_threshold else np.mean(distances)
        if abs(prev_error - mean_error) < threshold:
            print(f"ICP converged at iteration {iteration + 1}, Mean Error: {mean_error:.6f}")
            break
        prev_error = mean_error

        # Save transformation for debugging
        steps.append(transformation)

    return transformation

def estimate_normals(points, k=10):
    """
    Estimate surface normals for each point using PCA on its k nearest neighbors.
    Works for both 2D and 3D point sets.
    
    Args:
        points (ndarray): N x 2 or N x 3 array of point coordinates.
        k (int): Number of neighbors used for local PCA.
    
    Returns:
        normals (ndarray): N x 2 or N x 3 array of unit normals.
    """
    n_points, dim = points.shape
    tree = KDTree(points)
    normals = np.zeros_like(points)

    for i in range(n_points):
        # 1) Query k nearest neighbors of point i
        idx = tree.query(points[i], k)[1]
        neighbors = points[idx]

        # 2) Mean-center the neighbors
        neighbors_mean = np.mean(neighbors, axis=0)
        centered = neighbors - neighbors_mean
        
        # 3) PCA: Covariance matrix + SVD
        cov = centered.T @ centered
        # The last column of V (or Vt[-1]) is the eigenvector corresponding to the smallest eigenvalue
        # which is typically the normal direction for a surface/curve
        _, _, Vt = np.linalg.svd(cov)
        
        # 4) Extract the normal
        normal = Vt[-1, :]

        # 5) Store and normalize the normal
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-9:
            normal /= norm_len

        normals[i] = normal

    return normals

def point_to_plane_icp(source, target, max_iterations=50, threshold=1e-6, k=5):
    """
    Point-to-Plane ICP algorithm for 2D and 3D point clouds.
    Minimizes the distance between a source point and the plane defined by a target point and its normal.
    Converges faster than Point-to-Point ICP on smooth or continuous surfaces.
    
    Internally estimates the target's normals via PCA, so you only need to pass `source` and `target`.
    
    Args:
        source (ndarray): Source point cloud (N x 2 or N x 3).
        target (ndarray): Target point cloud (M x 2 or M x 3).
        max_iterations (int): Maximum number of ICP iterations.
        threshold (float): Convergence threshold for the change in mean error.
        k (int): Number of neighbors used to compute each target point's normal.
    
    Returns:
        transformation (ndarray): Final transformation matrix
            - (3 x 3) for 2D
            - (4 x 4) for 3D
        steps (list): Contains (source_snapshot, correspondences) for each iteration.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Validate dimensions
    assert source.shape[1] == target.shape[1], "Source and target must have the same dimensionality."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D or 3D point clouds are supported."

    # Compute normals for the target
    normals = estimate_normals(target, k=k)
    
    # Build a KD-tree for nearest neighbor lookups in the target
    tree = cKDTree(target)

    transformation = np.eye(dim + 1)  # (3x3) for 2D, (4x4) for 3D
    steps = []

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # 1) Find nearest neighbors for each source point
        distances, correspondences = tree.query(source)
        
        # 2) Retrieve corresponding target points and normals
        target_corr = target[correspondences]
        target_normals = normals[correspondences]

        # 3) Compute error as the scalar projection of the difference onto target normals
        diff = target_corr - source
        errors = np.sum(diff * target_normals, axis=1)  # Project the difference onto the normals
        
        # 4) Build the Jacobian (A) and residual vector (b)
        if dim == 2:
            # For 2D, we solve for (tx, ty, theta)
            A = np.zeros((len(source), 3))  # shape: [N, 3]
            # Partial derivatives wrt tx, ty
            A[:, 0:2] = target_normals  
            # Partial derivative wrt theta (rotation) => cross product in 2D
            # cross((x, y), normal_x, normal_y) => x * ny - y * nx
            A[:, 2] = source[:, 0] * target_normals[:, 1] - source[:, 1] * target_normals[:, 0]
        else:
            # For 3D, we solve for (tx, ty, tz, rx, ry, rz) using small-angle approximation
            A = np.zeros((len(source), 6))  
            # Partial derivatives wrt translation => normal
            A[:, :3] = target_normals
            # Partial derivatives wrt rotation => cross(source, normal)
            A[:, 3:] = np.cross(source, target_normals)

        b = errors

        # (Optional) Weights can be used to reduce outliers' impact
        weights = 1.0 / (distances + 1e-8)
        A_weighted = A * weights[:, np.newaxis]
        b_weighted = b * weights

        # 5) Solve the linear system for the incremental transform parameters
        x, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)

        # 6) Build the incremental transformation
        if dim == 2:
            # 2D rotation from x[2] and translation from x[:2]
            tx, ty, theta = x
            theta = theta % (2 * np.pi)  # normalize angle
            R_inc = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            t_inc = np.array([tx, ty])
            
            current_transformation = np.eye(3)
            current_transformation[:2, :2] = R_inc
            current_transformation[:2, 2] = t_inc

            # Update transformation matrix
            transformation = current_transformation @ transformation

            # Apply to source
            source = (R_inc @ source.T).T + t_inc

        else:
            # 3D uses Rodrigues-like formula for small-angle updates
            tx, ty, tz, rx, ry, rz = x
            t_inc = np.array([tx, ty, tz])

            # Rotation axis is (rx, ry, rz)
            theta = np.linalg.norm([rx, ry, rz])
            if theta > 1e-12:
                k = np.array([rx, ry, rz]) / theta
                K = np.array([
                    [0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]
                ])
                R_inc = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            else:
                R_inc = np.eye(3)

            current_transformation = np.eye(4)
            current_transformation[:3, :3] = R_inc
            current_transformation[:3, 3] = t_inc

            # Update transformation matrix
            transformation = current_transformation @ transformation

            # Apply to source
            source = (R_inc @ source.T).T + t_inc

        # Save iteration data (for debugging or visualization)
        #steps.append((source.copy(), correspondences))

        # 7) Compute mean error and check for convergence
        mean_error = np.mean(np.abs(errors))
        # logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")
        if abs(previous_error - mean_error) < threshold:
            logger.info("Convergence achieved.")
            break

        #previous_error = mean_error

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
    tree = cKDTree(target)

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

def estimate_local_covariances(points, k=10):
    """
    Compute a local covariance for each point using k-nearest neighbors.
    Works for both 2D and 3D point sets.
    
    Args:
        points (ndarray): Shape (N, 2) or (N, 3). 2D or 3D point cloud.
        k (int): Number of neighbors (excluding the point itself) to use for local PCA.
    
    Returns:
        covariances (ndarray): Shape (N, dim, dim). Each slice is the local covariance matrix
                               for one point in the cloud.
    """
    n_points, dim = points.shape
    # Build a KD-tree for neighbor queries
    tree = cKDTree(points)
    covariances = np.zeros((n_points, dim, dim))

    for i in range(n_points):
        # Query k+1 neighbors (including the point itself)
        # if you want exactly k neighbors besides the point, use k+1
        indices = tree.query(points[i], k+1)[1]
        neighbors = points[indices]

        # Mean-center the neighbors
        neighbors_mean = neighbors.mean(axis=0)
        centered = neighbors - neighbors_mean
        
        # Covariance = (1/(k)) * sum( (p_i)(p_i)^T ), ignoring 1/(k-1) vs 1/k factor for a small set
        cov = (centered.T @ centered) / max(1, (len(neighbors) - 1))
        covariances[i] = cov

    return covariances


def generalized_icp(source, target, max_iterations=50, threshold=1e-6, k=2):
    """
    Generalized ICP algorithm for 2D and 3D point clouds.
    Incorporates point-wise covariance matrices to account for local surface uncertainty.
    Internally estimates covariance for each point in source and target via PCA on k-nearest neighbors.
    
    Args:
        source (ndarray): Source point cloud (N x 2) or (N x 3).
        target (ndarray): Target point cloud (M x 2) or (M x 3).
        max_iterations (int): Maximum number of GICP iterations.
        threshold (float): Convergence threshold for the change in mean error.
        k (int): Number of neighbors used to compute local covariance for each point.
    
    Returns:
        transformation (ndarray):
            - (3 x 3) for 2D,
            - (4 x 4) for 3D,
          representing the final transformation.
        steps (list): [(source_snapshot, correspondences), ...] for each iteration.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Validate input dimensionality
    assert source.shape[1] == target.shape[1], "Source and target must have the same dimensionality."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D or 3D point clouds are supported."

    # --- 1) Estimate local covariances for source and target
    logger.info("Estimating local covariances for source and target...")
    source_cov = estimate_local_covariances(source, k=k)
    target_cov = estimate_local_covariances(target, k=k)

    # --- 2) Build KD-tree for the target
    tree = cKDTree(target)

    # --- 3) Initialize the transformation (3x3 if 2D, 4x4 if 3D)
    transformation = np.eye(dim + 1)
    steps = []

    prev_mean_error = float('inf')

    # --- 4) Main iteration loop
    for iteration in range(max_iterations):
        # 4a) Nearest neighbor matching
        distances, correspondences = tree.query(source)
        
        # 4b) Extract corresponding target points and covariances
        target_corr = target[correspondences]
        target_cov_corr = target_cov[correspondences]
        source_cov_corr = source_cov  # same index as source, since source[i] matched target_corr[i]

        # 4c) Form the combined covariance = source_cov + target_cov
        #     plus a small regularization to avoid numerical issues
        combined_cov = source_cov_corr + target_cov_corr + np.eye(dim) * 1e-8
        
        # 4d) Invert the combined covariance
        #     Using a loop since shape is (N, dim, dim)
        inv_combined_cov = np.zeros_like(combined_cov)
        for i in range(len(combined_cov)):
            try:
                inv_combined_cov[i] = np.linalg.inv(combined_cov[i])
            except np.linalg.LinAlgError:
                # Add a small diagonal for numerical stability
                stable_cov = combined_cov[i] + np.eye(dim) * 1e-8
                inv_combined_cov[i] = np.linalg.inv(stable_cov)

        # 4e) Weighted difference
        diff = target_corr - source  # shape (N, dim)
        # Weighted difference: multiply diff by inverse covariance
        # => (N, dim, dim) @ (N, dim, 1) => (N, dim)
        weighted_diff = np.einsum('nij,nj->ni', inv_combined_cov, diff)

        # 4f) Compute cross-covariance
        #     We'll handle it similarly to standard ICP, but with weighting
        source_centroid = source.mean(axis=0)
        target_centroid = target_corr.mean(axis=0)
        source_demean = source - source_centroid
        target_demean = target_corr - target_centroid

        # Weighted cross-covariance, W = sum( s_i^T * inv_cov_i * t_i ), approximated with Einstein summation
        # A simpler approach is W = S^T ( WeightedDiff ), but we have to handle the demean properly
        # Here we'll do a partial weighting:
        W = (source_demean.T @ weighted_diff)

        # 4g) Rotation from SVD
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure a proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = target_centroid - R @ source_centroid

        # 4h) Build incremental transformation
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Update overall transformation
        transformation = current_transformation @ transformation

        # 4i) Apply transformation to source
        source = (R @ source.T).T + t

        # 4j) Save iteration data
        steps.append((source.copy(), correspondences.copy()))

        # 4k) Compute mean error (Euclidean) for convergence
        mean_error = np.mean(np.linalg.norm(diff, axis=1))
        #logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

        # 4l) Check if improvement is below threshold
        if abs(prev_mean_error - mean_error) < threshold:
            logger.info(f"Convergence achieved at iteration {iteration + 1}.")
            break

        prev_mean_error = mean_error

    return transformation

def robust_icp(source, target, max_iterations=50, threshold=1e-6, 
               loss_function="tukey", loss_param=1.0, verbose=False):
    """
    Robust Iterative Closest Point (ICP) algorithm with robust loss functions (Huber, Tukey).
    This variant reduces sensitivity to outliers by applying **adaptive loss weights**.

    Args:
        source (ndarray): Source point cloud (Nx2 or Nx3).
        target (ndarray): Target point cloud (Mx2 or Mx3).
        max_iterations (int): Maximum number of iterations.
        threshold (float): Convergence threshold based on weighted mean error.
        loss_function (str): Robust loss function, either "huber" or "tukey".
        loss_param (float): Loss function parameter (delta for Huber, c for Tukey).
        verbose (bool): If True, prints debug logs.

    Returns:
        transformation (ndarray): Final transformation matrix (3x3 for 2D, 4x4 for 3D).
        steps (list): List of intermediate source points and correspondences.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    assert source.shape[0] > 0, "Point clouds must not be empty."
    dim = source.shape[1]
    assert dim in [2, 3], "Only 2D and 3D point clouds are supported."
    assert loss_param > 0, "Loss parameter must be positive."

    transformation = np.eye(dim + 1)  # 3x3 for 2D, 4x4 for 3D
    steps = []

    # KD-Tree for efficient nearest neighbor search
    tree = cKDTree(target)  # Faster than KDTree

    previous_error = float('inf')

    for iteration in range(max_iterations):
        # Step 1: Find nearest neighbors
        distances, correspondences = tree.query(source)
        target_matched = target[correspondences]

        # Step 2: Compute residuals
        residuals = target_matched - source
        norm_residuals = np.linalg.norm(residuals, axis=1)

        # Step 3: Apply robust loss function
        if loss_function == "huber":
            weights = np.where(
                norm_residuals <= loss_param,
                1.0,
                loss_param / norm_residuals
            )
        elif loss_function == "tukey":
            weights = np.where(
                norm_residuals <= loss_param,
                (1 - (norm_residuals / loss_param) ** 2) ** 2,
                0
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # Handle cases where all weights are zero
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            logger.warning(f"Iteration {iteration + 1}: All weights are zero. Skipping update.")
            continue  # Skip this iteration instead of breaking

        # Step 4: Compute weighted centroids
        source_centroid = np.average(source, axis=0, weights=weights)
        target_centroid = np.average(target_matched, axis=0, weights=weights)

        # Step 5: Demean the points
        source_demean = source - source_centroid
        target_demean = target_matched - target_centroid

        # Step 6: Compute weighted cross-covariance matrix
        W = (source_demean * weights[:, None]).T @ target_demean

        # Step 7: Compute optimal rotation using Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure a valid rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Step 8: Compute translation
        t = target_centroid - R @ source_centroid

        # Step 9: Construct transformation matrix
        current_transformation = np.eye(dim + 1)
        current_transformation[:dim, :dim] = R
        current_transformation[:dim, dim] = t

        # Step 10: Update cumulative transformation
        transformation = current_transformation @ transformation

        # Step 11: Transform source points
        source = (R @ source.T).T + t

        # Store intermediate results
        steps.append((source.copy(), correspondences.copy()))

        # Step 12: Compute weighted mean error and check convergence
        mean_error = np.mean(weights * norm_residuals)
        if verbose:
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
    tree = cKDTree(target)

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

def sparse_icp(source, target, sparsity=0.9, max_iterations=50, threshold=1e-6):
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
    tree = cKDTree(target_sparse)

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
        #logger.info(f"Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

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
        tree = cKDTree(target_downsampled)

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
            #logger.info(f"Scale {scale}, Iteration {iteration + 1}: Mean Error = {mean_error:.6f}")

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
    tree_source = cKDTree(source)
    tree_target = cKDTree(target)

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
    tree_target = cKDTree(target)

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
    tree = cKDTree(target)

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

def coarse_to_fine_icp(source, target, levels=[8, 4, 2, 1], max_iterations=50, threshold=1e-6):
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
        current_transformation = robust_icp(source_ds, target_ds, max_iterations=max_iterations, threshold=threshold)

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