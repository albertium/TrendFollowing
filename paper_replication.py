
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate, optimize


def generate_series(S0, mu1, mu2, sig, lmbda, r, T, N=10000, M=1000):
    tau = np.random.exponential(1 / lmbda, N)
    dt = T / M
    St = np.zeros([N, M])
    St[:, 0] = S0
    for i in range(1, M):
        t = (i + 1) * dt
        mu = mu1 + (t > tau) * (mu2 - mu1)
        St[:, i] = (np.random.normal(loc=mu * dt, scale=sig * np.sqrt(dt), size=N) + 1) * St[:, i - 1]
    return St, tau


def moving_average_strategy(St, delta, dt):
    order = int(delta / dt)
    tmp = np.hstack([np.zeros([St.shape[0], 1]), np.cumsum(St, axis=1)])
    moving_averages = (tmp[:, order:] - tmp[:, : -order]) / order
    return St[:, order - 1:] > moving_averages


def evaluate_strategy(St, policy, r, dt):
    St = St[:, -policy.shape[1]:]
    policy = policy[:, :-1]
    risk_free = np.exp(r * dt)
    return np.cumsum(np.mean(np.log(policy * (St[:, 1:] / St[:, :-1] - risk_free) + risk_free), axis=0))


def evaluate_strategies(St, policies, r, dt):
    length = np.min([policy.shape[1] for policy in policies])
    new_policies = [policy[:, -length:] for policy in policies]
    performances = []
    for x in new_policies:
        performances.append(evaluate_strategy(St, x, r, dt))
    return performances


def get_pstar(mu1, mu2, sig, lmbda):
    beta = 2 * lmbda * sig ** 2 / (mu1 - mu2) ** 2

    def integrand(s):
        return (1 - 2 * s) * np.exp(-beta / s) / np.power(1 - s, 2 + beta) / np.power(s, 2 - beta)

    target = integrate.quad(integrand, 0, 0.5)[0]
    return optimize.brentq(lambda x: integrate.quad(integrand, x, 0.5)[0] - target, 0.5, 0.999)


def get_posterior(St, S0, mu1, mu2, sig, lmbda, dt):
    drift = ((mu2 - mu1) ** 2 + 2 * (mu2 - mu1) * (mu1 - sig ** 2 / 2)) / 2 / sig ** 2
    Ft = np.zeros_like(St)
    integral = 0
    for idx in range(1, St.shape[1]):
        t = idx * dt
        Lt = np.power(St[:, idx] / S0, (mu2 - mu1) / sig ** 2) * np.exp(-drift * t)
        tmp = np.exp(lmbda * t) * Lt
        integral += dt / tmp
        tmp = lmbda * tmp * integral
        Ft[:, idx] = tmp / (1 + tmp)
    return Ft


def karataz_strategy(St, S0, mu1, mu2, sig, lmbda, dt):
    threshold = get_pstar(mu1, mu2, sig, lmbda)
    Ft = get_posterior(St, S0, mu1, mu2, sig, lmbda, dt)
    # plt.plot(Ft.T)
    return np.minimum(np.cumsum(Ft > threshold, axis=1), 1)


def optimal_strategy(St, change_points, dt):
    time_points = np.arange(St.shape[1]) * dt
    return time_points > change_points[:, None]


S0 = 100
mu1 = -0.2
mu2 = 0.2
sig = 0.15
lmbda = 0.5
r = 0.0
T = 2
N = 20000
M = 1000
dt = T / M

St, change_points = generate_series(S0=S0, mu1=mu1, mu2=mu2, sig=sig, lmbda=lmbda, r=r, T=T, N=N, M=M)
# metrics = []
# deltas = np.linspace(0.05, 1, 10)
# for idx, delta in enumerate(deltas):
#     policy = moving_average_strategy(St, delta, dt)
#     policy = karataz_strategy(St, S0, mu1, mu2, sig, lmbda, dt)
    # metrics.append(evaluate_strategy(St, policy, r, dt)[-1])
    # print(idx)
# plt.plot(deltas, metrics)

policies = []
policies.append(karataz_strategy(St, S0, mu1, mu2, sig, lmbda, dt))
policies.append(moving_average_strategy(St, 0.5, dt))
policies.append(optimal_strategy(St, change_points, dt))
performances = evaluate_strategies(St, policies, r, dt)
pd.DataFrame(np.array(performances).T).plot()
plt.show()