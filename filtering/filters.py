import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.monte_carlo import systematic_resample


# Kalman Filter
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
        weights += 1.e - 300
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
    true_states, measurements, filtered_states = apply_extended_kalman_filter()
    plot_results(true_states, measurements, filtered_states, 'Extended Kalman Filter')

    # Unscented Kalman Filter
    true_states, measurements, filtered_states = apply_unscented_kalman_filter()
    plot_results(true_states, measurements, filtered_states, 'Unscented Kalman Filter')

    # Particle Filter
    true_states, measurements, filtered_states = apply_particle_filter()
    plot_results(true_states, measurements, filtered_states, 'Particle Filter')
