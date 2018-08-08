
from scipy import integrate, optimize
import numpy as np


class KaratzaFactory:
    """
    arithmetic Brownian motion change point detection
    """
    def __init__(self, S0, theta, sig, lmbda, dt):
        self.S0 = S0
        self.theta = theta
        self.sig = sig
        self.sig2 = sig ** 2
        self.lmbda = lmbda
        self.dt = dt
        self.t = dt

        self.Lambda = 2 * lmbda * self.sig2 / theta ** 2
        self.integral = 0  # for Ft
        self.prev_value = 0  # for Ft
        self.pstar = self.get_pstar()
        self.threshold = self.pstar / (1 - self.pstar)

    def integrand(self, s):
        return (1 - 2 * s) * np.exp(-self.Lambda / s) / np.power(1 - s, 2 + self.Lambda) / np.power(s, 2 - self.Lambda)

    def get_pstar(self) -> float:
        target = integrate.quad(self.integrand, 0, 0.5)[0]
        return optimize.brentq(lambda x: integrate.quad(self.integrand, x, 0.5)[0] - target, 0.5, 0.999)

    def reset(self):
        self.integral = 0
        self.prev_value = 0
        self.t = 0

    def check(self, Xt):
        value = np.exp(-self.theta / self.sig2 * Xt + (self.theta ** 2 / 2 / self.sig2 - self.lmbda) * self.t)
        self.integral += self.lmbda * value * self.dt
        self.t += self.dt
        return self.integral > self.threshold * value

    def generate(self, N2=1000):
        change_point = np.random.exponential(1 / self.lmbda, 1)
        N1 = int(np.round(change_point / self.dt)[0])
        dXt = np.random.normal(0, self.sig * np.sqrt(self.dt), N1)
        dXt = np.concatenate([dXt, np.random.normal(self.theta * self.dt, self.sig * np.sqrt(self.dt), N2)])
        return N1, np.cumsum(dXt)
