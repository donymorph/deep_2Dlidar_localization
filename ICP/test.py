import numpy as np
from scipy.spatial import cKDTree

def build_voxel_grid(points, voxel_size=0.5):
    """
    Build a simple voxel grid over 2D points.
    Returns a dict voxel_id -> list_of_points.
    """
    # Quantize points into integer voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    voxel_map = {}
    for idx, voxel_id in enumerate(voxel_indices):
        key = tuple(voxel_id)  # so it can be used as dict key
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(points[idx])
    return voxel_map

def compute_voxel_stats(voxel_map):
    """
    For each voxel, compute mean and covariance of the points.
    Returns voxel_id -> (mean, covariance, inverse_cov, weight).
    """
    voxel_stats = {}
    for key, pts in voxel_map.items():
        pts_arr = np.array(pts)
        mean = np.mean(pts_arr, axis=0)
        centered = pts_arr - mean
        cov = centered.T @ centered / max(1, (len(pts_arr) - 1))
        
        # Regularize small cov for numerical stability
        cov += 1e-5 * np.eye(2)
        
        inv_cov = np.linalg.inv(cov)
        weight = len(pts_arr)
        voxel_stats[key] = (mean, cov, inv_cov, weight)
    return voxel_stats

def transform_point_2d(point, x, y, theta):
    """
    Apply a 2D transformation [x, y, theta] to point (px, py).
    """
    px, py = point
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Rotate then translate
    rx = cos_t * px - sin_t * py + x
    ry = sin_t * px + cos_t * py + y
    return np.array([rx, ry])

def ndt_score_and_deriv_2d(source_points, voxel_stats, transform, voxel_size=0.5):
    """
    Compute the negative log-likelihood score + gradient w.r.t. transform parameters [x, y, theta].
    Simplified for illustration; a real NDT uses a more robust weighting, Hessian, etc.
    """
    x, y, theta = transform
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Precompute partial derivatives of rotation
    # d(R*x)/d(theta)
    # if R(theta) = [[cosθ, -sinθ],[sinθ, cosθ]], 
    # derivative wrt θ = [[-sinθ, -cosθ],[cosθ, -sinθ]]

    score = 0.0
    grad = np.zeros(3)  # d(score)/dx, d(score)/dy, d(score)/dθ

    for pt in source_points:
        # 1) Transform point
        px, py = pt
        rx = cos_t * px - sin_t * py + x
        ry = sin_t * px + cos_t * py + y

        # 2) Find which voxel it falls in
        vid = tuple(np.floor(np.array([rx, ry])/voxel_size).astype(int))

        if vid not in voxel_stats:
            # If outside, no contribution
            continue

        mean, cov, inv_cov, weight = voxel_stats[vid]

        # 3) Negative log-likelihood of a Gaussian
        # E = 0.5 * (rx - mean_x, ry - mean_y)^T * inv_cov * (rx - mean_x, ry - mean_y)
        diff = np.array([rx, ry]) - mean
        e = 0.5 * diff @ inv_cov @ diff
        score += e

        # 4) Gradient of that distance
        # dE/d(rx, ry) = diff^T * inv_cov
        # Then chain rule for d(rx, ry)/dx, d(rx, ry)/dy, d(rx, ry)/dθ
        #   rx = cosθ*px - sinθ*py + x
        #   ry = sinθ*px + cosθ*py + y

        dE_drx = diff @ inv_cov[0]
        dE_dry = diff @ inv_cov[1]

        # partial wrt x
        dE_dx = dE_drx * 1.0 + dE_dry * 0.0  # rx partial wrt x is 1, ry partial wrt x is 0
        # partial wrt y
        dE_dy = dE_drx * 0.0 + dE_dry * 1.0  # rx partial wrt y is 0, ry partial wrt y is 1
        # partial wrt theta
        # rx partial wrt θ = -sinθ*px - cosθ*py
        # ry partial wrt θ =  cosθ*px - sinθ*py
        dE_dtheta = dE_drx * (-sin_t*px - cos_t*py) + dE_dry * (cos_t*px - sin_t*py)

        grad[0] += dE_dx
        grad[1] += dE_dy
        grad[2] += dE_dtheta

    # Return negative log-likelihood (the smaller the better), so for gradient-based
    # we might invert sign or handle carefully.  We'll do a direct approach: we want to minimize 'score'
    # so we'll treat 'score' as the objective function to minimize with gradient 'grad'.
    return score, grad

def ndt_registration_2d(source, target, voxel_size=0.5, 
                        max_iterations=50, step_size=0.1, 
                        convergence_tol=1e-4):
    voxel_map = build_voxel_grid(target, voxel_size)
    voxel_stats = compute_voxel_stats(voxel_map)

    transform = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
    prev_score = float('inf')

    for iteration in range(max_iterations):
        score, grad = ndt_score_and_deriv_2d(source, voxel_stats, transform, voxel_size)

        # ---------- Simple line search or step clamp ----------
        alpha = step_size
        # Take a negative step along the gradient => we want to MINIMIZE 'score'
        proposed = transform - alpha * grad

        new_score, _ = ndt_score_and_deriv_2d(source, voxel_stats, proposed, voxel_size)

        # If new_score is bigger, try smaller alpha
        while new_score > score and alpha > 1e-6:
            alpha *= 0.5
            proposed = transform - alpha * grad
            new_score, _ = ndt_score_and_deriv_2d(source, voxel_stats, proposed, voxel_size)
        # ------------------------------------------------------

        transform = proposed

        # Convergence check
        if abs(prev_score - new_score) < convergence_tol:
            print(f"NDT converged at iteration {iteration+1}, score={new_score:.5f}")
            break
        prev_score = new_score

    return transform

# ------------------ EXAMPLE USAGE ------------------
if __name__ == "__main__":
    # Generate a synthetic dataset
    np.random.seed(42)
    # target: random points in 2D
    target_points = np.random.rand(200, 2) * 10.0  # 200 points in [0..10] x [0..10]

    # Let's transform target by some known transform to create source
    # e.g., [tx, ty, theta] = [2, -1, 30 degrees]
    true_transform = np.array([2.0, -1.0, np.radians(30.0)])
    
    def apply_transform_2d(points, transform):
        x, y, theta = transform
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]])
        return points @ R.T + np.array([x, y])

    source_points = apply_transform_2d(target_points, true_transform)
    # (Optional) Add some Gaussian noise
    source_points += np.random.normal(scale=0.05, size=source_points.shape)

    # Try to recover transform using NDT
    estimated_transform = ndt_registration_2d(source_points, target_points,
                                             voxel_size=0.1,
                                             max_iterations=10,
                                             step_size=0.001,
                                             convergence_tol=1e-6)
    print("True transform:", true_transform)
    print("Estimated transform:", estimated_transform)
