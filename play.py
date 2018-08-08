
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate


def find_root(f, target, x0, x1):
    f0 = f(x0) - target
    f1 = f(x1) - target

    if f0 * f1 > 0:
        raise ValueError("Initial values not proper")

    if f0 > f1:
        x0, x1, = x1, x0

    while abs(x0 - x1) > 1e-8:
        new_x = (x0 + x1) / 2
        new_f = f(new_x) - target
        if new_f < 0:
            x0 = new_x
        else:
            x1 = new_x

    return x1


class KaratzaFactory:
    def __init__(self, S0, mu1, mu2, sig, lmbda):
        self.S0 = S0
        self.mu1 = mu1
        self.mu2 = mu2
        self.sig = sig
        self.sig2 = sig ** 2
        self.lmbda = lmbda

        self.beta = 2 * lmbda * self.sig2 / (mu1 - mu2) ** 2
        self.const1 = (mu2 - mu1) / self.sig2  # for Lt
        self.const2 = -((mu2 - mu1) ** 2 + 2 * (mu2 - mu1) * (mu1 - self.sig2 / 2)) / 2 / self.sig2  # for Lt
        self.integral = 0  # for Ft
        self.prev_t = 0  # for Ft
        self.prev_value = 0  # for Ft

    def integrand(self, s):
        return (1 - 2 * s) * np.exp(-self.beta / s) / np.power(1 - s, 2 + self.beta) * np.power(s, 2 - self.beta)

    def get_pstar(self):
        target = integrate.quad(self.integrand, 0, 0.5)[0]
        return find_root(lambda x: integrate.quad(self.integrand, x, 0.5)[0], target, 0.5, 0.75)

    def Lt(self, St, t):
        return np.power(St / self.S0, self.const1) * np.exp(self.const2 * t)

    def Ft(self, St, t):
        tmp = np.exp(self.lmbda * t) * self.Lt(St, t)
        self.integral += (self.prev_value + 1 / tmp) / 2 * (t - self.prev_t)
        val = self.lmbda * tmp * self.integral
        self.prev_t = t
        self.prev_value = 1 / tmp
        return 1 - 1 / (1 + val)

    def generate(self, T, N1, N2, M):
        dt = T / (N1 + N2)
        dB1 = np.random.normal((self.mu1 - 0.5 * self.sig2) * dt, self.sig * np.sqrt(dt), [M, N1])
        dB2 = np.random.normal((self.mu2 - 0.5 * self.sig2) * dt, self.sig * np.sqrt(dt), [M, N2])
        dB = np.hstack([dB1, dB2])
        B = np.cumsum(dB, axis=1)
        return self.S0 * np.exp(B)


if __name__ == "__main__":
    S0 = 100
    mu1 = 0
    mu2 = 20
    sig = 0.3
    lmbda = 0.1
    T = 1
    N1 = 200
    N2 = 200
    M = 10000

    factory = KaratzaFactory(S0, mu1, mu2, sig, lmbda)
    # print(factory.get_pstar())
    series = factory.generate(1, N1, N2, M)
    # print(series.shape)
    # dt = T / (N1 + N2)
    # t = dt
    # post_prob = []
    # for St in series[0]:
    #     post_prob.append(factory.Ft(St, t))
    #     t += dt
    # post_prob = np.array(post_prob)
    #
    # threshold = factory.get_pstar()
    # detected = np.argmax(post_prob > threshold)
    # print(detected)

    # data = pd.DataFrame({"orig": series[0]})
    # data["ret"] = (data.orig / data.orig.shift(1)) - 1
    # data.orig.plot()
    # data.ret.plot(secondary_y=True)
    # plt.show(block=True)

    # test changepoint
    ret = (series[0, 1:] / series[0, :-1]) - 1
    import bayesian_changepoint_detection.online_changepoint_detection as oncd
    from functools import partial

    R, maxes = oncd.online_changepoint_detection(ret, partial(oncd.constant_hazard, 50), oncd.StudentT(0.1, .01, 1, 0))
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(ret)
    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    sparsity = 5  # only plot every fifth data for faster display
    ax.pcolor(np.array(range(0, len(R[:, 0]), sparsity)),
              np.array(range(0, len(R[:, 0]), sparsity)),
              -np.log(R[0:-1:sparsity, 0:-1:sparsity]),
              cmap=cm.Greys, vmin=0, vmax=30)
    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    Nw = 10
    ax.plot(R[Nw, Nw:-1])
    plt.show()