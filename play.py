
import time
from detection import KaratzaFactory
from matplotlib import pyplot as plt
import numpy as np


def wrapper(factory):
    change, series = factory.generate()
    factory.reset()
    for idx, Xt in enumerate(series):
        if factory.check(Xt):
            return change, idx

if __name__ == "__main__":
    S0 = 100
    theta = 0.45
    sig = 0.3
    lmbda = 0.2
    dt = 0.01
    factory = KaratzaFactory(S0, theta, sig, lmbda, dt)

    diff = []
    too_early = 0
    start = time.clock()
    for _ in range(1000):
        a, b = wrapper(factory)
        diff.append(abs(a - b))
        too_early += b < a
    print(np.mean(diff))
    print(too_early)
    print(time.clock() - start)
