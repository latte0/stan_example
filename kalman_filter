import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter():
    def __init__(self, A, B, C, x, u, z, epsilon, mean_epsilon, R, delta, mean_delta, Q):
        (_, self.n, self.t) = A.shape
        (_, self.m, _)      = B.shape
        (self.k, _, _)       = C.shape

        self.A = A
        self.B = B
        self.C = C
        self.x = x
        self.u = u
        self.z = z
        self.epsilon = epsilon
        self.mean_epsilon = mean_epsilon
        self.R = R
        self.delta = delta
        self.mean_delta = mean_delta
        self.Q = Q

        self.mu = np.zeros(shape=(self.t, self.n))
        self.sigma = np.zeros(shape=(self.n, self.n, self.t))

        self.mu[0, :] = np.ones(shape=self.mu[0, :].shape) * 10
        self.sigma[:,:,0] = np.eye(self.n) * 1000

        self.mu_bar = np.zeros(shape=self.mu.shape)
        self.sigma_bar = np.zeros(shape=self.sigma.shape)

        self.K = np.zeros(shape=(self.n, self.k, self.t))

    def execute(self):
        for i in range(1, self.t ):
            # 予測
            self.mu_bar[i, :] = np.dot(self.A[:,:,i], self.mu[i - 1, :]) + np.dot(self.B[:,:,i], self.u[i,:])
            self.sigma_bar[:,:,i] = np.dot(np.dot(self.A[:,:,i], self.sigma[:,:,i - 1]), self.A[:,:,i].T) + self.R[:,:,i]

            # 計測更新
            self.K[:,:,i] = np.dot(
                np.dot(self.sigma_bar[:,:,i], self.C[:,:,i].T),
                np.linalg.inv(
                    np.dot(np.dot(self.C[:,:,i], self.sigma_bar[:,:,i]), self.C[:,:,i].T) + self.Q[:,:,i]
                )
            )

            self.mu[i, :] = self.mu_bar[i, :] + np.dot(self.K[:,:,i], self.z[i, :] - np.dot(self.C[:,:,i], self.mu_bar[i, :].T))
            self.sigma[:,:,i] =np.dot((np.eye(self.n) - np.dot(self.K[:,:,i], self.C[:,:,i])), self.sigma_bar[:,:,i])
