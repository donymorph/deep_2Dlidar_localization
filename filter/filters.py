import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.monte_carlo import systematic_resample

#############################
# modified filters you can use them in visualize_filters.py script
#############################
def apply_kalman_filter_3d(measurements):
    """
    Applies a Kalman filter to a series of 3D measurements.
    Input:
        measurements: NumPy array of shape (N, 3)
    Returns:
        filtered_states: NumPy array of shape (N, 3)
    """
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.F = np.eye(3)
    kf.H = np.eye(3)
    kf.x = np.zeros((3, 1))
    kf.P = np.eye(3)
    kf.Q = np.eye(3) * 0.01  # Process noise covariance
    kf.R = np.eye(3) * 0.1   # Measurement noise covariance

    filtered_states = []
    for z in measurements:
        kf.predict()
        kf.update(z.reshape(3, 1))
        filtered_states.append(kf.x.flatten())
    return np.array(filtered_states)

def apply_extended_kalman_filter_3d(measurements):
    """
    Applies Extended Kalman Filter on 3D measurements [pos_x, pos_y, orientation_z].
    Returns the filtered states.
    """
    def f(x, dt):
        # State transition: Predicts the next state based on previous state (position and orientation).
        return np.array([x[0], x[1], x[2]])  # Position and orientation don't change in this simple model

    def h(x):
        # Measurement function: Directly measures position and orientation
        return np.array([x[0], x[1], x[2]])
    def HJacobian(x):
        # Jacobian of the measurement function
        return np.eye(3)

    def Hx(x):
        # Measurement function
        return np.array([x[0], x[1], x[2]])
    # Initialize the EKF
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=3)
    ekf.x = np.array([measurements[0, 0], measurements[0, 1], measurements[0, 2]])  # Initial state (position, orientation)
    ekf.P = np.eye(3) * 0.1  # Initial uncertainty
    ekf.F = np.eye(3)  # State transition matrix
    ekf.H = np.eye(3)  # Measurement matrix
    ekf.R = np.eye(3) * 0.1  # Measurement noise
    ekf.Q = np.eye(3) * 0.01  # Process noise

    filtered_states = []
    for z in measurements:
        ekf.x = f(ekf.x, 0.1)  # Apply state transition function
        ekf.predict()  # Predict the next state
        ekf.update(z, HJacobian=HJacobian, Hx=Hx)  # Update the filter with the measurement
        filtered_states.append(ekf.x[:3])  # Store the filtered position and orientation

    return np.array(filtered_states)


def apply_unscented_kalman_filter_3d(measurements):
    """
    Applies Unscented Kalman Filter on 3D measurements [pos_x, pos_y, orientation_z].
    Returns the filtered states.
    """
    def f(x, dt):
        # State transition: Predicts the next state based on previous state (position and orientation).
        return np.array([x[0], x[1], x[2]])  # Position and orientation don't change in this simple model

    def h(x):
        # Measurement function: Directly measures position and orientation
        return np.array([x[0], x[1], x[2]])

    # Initialize the UKF
    points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=3, dt=0.1, fx=f, hx=h, points=points)
    ukf.x = np.array([measurements[0, 0], measurements[0, 1], measurements[0, 2]])  # Initial state (position, orientation)
    ukf.P = np.eye(3) * 0.1  # Initial uncertainty
    ukf.R = np.eye(3) * 0.1  # Measurement noise
    ukf.Q = np.eye(3) * 0.01  # Process noise

    filtered_states = []
    for z in measurements:
        ukf.predict()  # Predict the next state
        ukf.update(z)  # Update the filter with the measurement
        filtered_states.append(ukf.x[:3])  # Store the filtered position and orientation

    return np.array(filtered_states)

def apply_particle_filter_3d(measurements):
    """
    Applies Particle Filter on 3D measurements [pos_x, pos_y, orientation_z].
    Returns the filtered states.
    """
    def f(x):
        # Motion model with Gaussian noise for position and orientation
        return x + np.random.normal(0, 0.1, x.shape)

    def h(x):
        # Measurement function with added noise
        return x + np.random.normal(0, 1, x.shape)

    np.random.seed(0)
    particles = np.random.normal(0, 1, (1000, 3))  # Initialize particles for 3D
    weights = np.ones(1000) / 1000  # Initialize equal weights for all particles

    filtered_states = []
    for z in measurements:
        particles = f(particles)  # Apply motion model
        likelihood = np.exp(-np.sum((z - h(particles))**2, axis=1) / 2)  # Likelihood of each particle
        weights *= likelihood
        weights /= np.sum(weights)  # Normalize the weights

        # Resample if necessary
        if 1. / np.sum(weights ** 2) < 500:
            indexes = systematic_resample(weights)
            particles[:] = particles[indexes]
            weights.fill(1.0 / 1000)

        filtered_states.append(np.average(particles, axis=0, weights=weights))  # Take the weighted average

    return np.array(filtered_states)
############################################################################################################
# END OF MODIFIED FILTERS
############################################################################################################

# Basic filters yours
def apply_kalman_filter(measurements):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[measurements[0]]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.P = np.array([[1.]])
    kf.R = np.array([[1.]])
    kf.Q = np.array([[0.01]])

    filtered_states = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        filtered_states.append(kf.x[0, 0])

    return filtered_states

# Extended Kalman Filter
def apply_extended_kalman_filter():
    def f(x, dt):
        return np.array([x[0] + dt * x[1], x[1]])

    def h(x):
        return np.array([x[0]])

    np.random.seed(0)
    true_states = np.zeros((2, 100))
    true_states[:, 0] = [0, 1]
    for i in range(1, 100):
        true_states[:, i] = f(true_states[:, i - 1], 0.1)
    measurements = true_states[0, :] + np.random.normal(0, 1, 100)

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([measurements[0], 0])
    ekf.F = np.array([[1, 0.1], [0, 1]])
    ekf.H = np.array([[1, 0]])
    ekf.P = np.eye(2)
    ekf.R = np.array([[1.]])
    ekf.Q = np.eye(2) * 0.01

    filtered_states = []
    for z in measurements:
        ekf.predict(f=f, args=(0.1,))
        ekf.update(z, HJacobian=lambda x: np.array([[1, 0]]), Hx=h)
        filtered_states.append(ekf.x[0])

    return true_states[0, :], measurements, filtered_states


# Unscented Kalman Filter
def apply_unscented_kalman_filter():
    def f(x, dt):
        return np.array([x[0] + dt * x[1], x[1]])

    def h(x):
        return np.array([x[0]])

    np.random.seed(0)
    true_states = np.zeros((2, 100))
    true_states[:, 0] = [0, 1]
    for i in range(1, 100):
        true_states[:, i] = f(true_states[:, i - 1], 0.1)
    measurements = true_states[0, :] + np.random.normal(0, 1, 100)

    points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=0.1, fx=f, hx=h, points=points)
    ukf.x = np.array([measurements[0], 0])
    ukf.P = np.eye(2)
    ukf.R = np.array([[1.]])
    ukf.Q = np.eye(2) * 0.01

    filtered_states = []
    for z in measurements:
        ukf.predict()
        ukf.update(z)
        filtered_states.append(ukf.x[0])

    return true_states[0, :], measurements, filtered_states


# Particle Filter
def apply_particle_filter():
    def f(x):
        return x + np.random.normal(0, 0.1)

    def h(x):
        return x + np.random.normal(0, 1)

    np.random.seed(0)
    true_states = np.zeros(100)
    true_states[0] = 0
    for i in range(1, 100):
        true_states[i] = f(true_states[i - 1])
    measurements = h(true_states)

    N = 1000
    particles = np.random.normal(0, 1, N)
    weights = np.ones(N) / N

    filtered_states = []
    for z in measurements:
        particles = f(particles)
        likelihood = np.exp(-(z - h(particles)) ** 2 / 2)
        weights *= likelihood
        #weights += 1.e - 300
        weights /= sum(weights)

        if 1. / np.sum(weights ** 2) < N / 2:
            indexes = systematic_resample(weights)
            particles[:] = particles[indexes]
            weights.fill(1.0 / N)

        filtered_states.append(np.average(particles, weights=weights))

    return true_states, measurements, filtered_states


# Plotting function
def plot_results(true_states, measurements, filtered_states, title):
    plt.plot(true_states, label='True States')
    plt.plot(measurements, label='Measurements (Noisy Predictions)', alpha=0.7)
    plt.plot(filtered_states, label='Filtered States', color='red')
    plt.title(title)
    plt.legend()
    plt.show()


# Generate sample noisy data
def generate_data():
    np.random.seed(0)
    true_states = np.linspace(0, 10, 100)
    measurements = true_states + np.random.normal(0, 1, 100)
    return true_states, measurements



# Main execution
if __name__ == "__main__":
    # Kalman Filter
    true_states, measurements = generate_data()
    filtered_states = apply_kalman_filter(measurements)
    plot_results(true_states, measurements, filtered_states, 'Kalman Filter')

    # Extended Kalman Filter
    #true_states, measurements, filtered_states = apply_extended_kalman_filter()
    #plot_results(true_states, measurements, filtered_states, 'Extended Kalman Filter')

    # Unscented Kalman Filter
    true_states, measurements, filtered_states = apply_unscented_kalman_filter()
    plot_results(true_states, measurements, filtered_states, 'Unscented Kalman Filter')

    # Particle Filter
    true_states, measurements, filtered_states = apply_particle_filter()
    plot_results(true_states, measurements, filtered_states, 'Particle Filter')