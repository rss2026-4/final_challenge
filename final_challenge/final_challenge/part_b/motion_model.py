import numpy as np

class MotionModel:

    def __init__(self, node):
        node.declare_parameter('deterministic', False)
        self.deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value
        self.node = node
        self.lin_noise_frac = 0.15
        self.ang_noise_frac = 0.15
        # self.noise_std = 0.15


    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of thera
                same size
        """

        dx, dy, dtheta = odometry
        cos_t = np.cos(particles[:, 2])
        sin_t = np.sin(particles[:, 2])

        particles[:, 0] += dx * cos_t - dy * sin_t
        particles[:, 1] += dx * sin_t + dy * cos_t
        particles[:, 2] += dtheta

        if not self.deterministic:
            N = particles.shape[0]
            dist = np.sqrt(dx * dx + dy * dy)
            particles[:, 0] += np.random.normal(0, self.lin_noise_frac * dist + 1e-4, N)
            particles[:, 1] += np.random.normal(0, self.lin_noise_frac * dist + 1e-4, N)
            particles[:, 2] += np.random.normal(0, self.ang_noise_frac * abs(dtheta) + 1e-4, N)
            # particles[:, 0] += np.random.normal(0, self.noise_std, N)
            # particles[:, 1] += np.random.normal(0, self.noise_std, N)
            # particles[:, 2] += np.random.normal(0, self.noise_std, N)
            

        return particles