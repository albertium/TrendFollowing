
import time
from detection import KaratzaFactory
import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
from matplotlib import pyplot as plt
import numpy as np


def wrapper(factory, N2=1000):
    change, series = factory.generate(N2)
    factory.reset()
    for idx, Xt in enumerate(series):
        if factory.check(Xt):
            return change, idx, series


def wrapper1(factory, N2=1000):
    change, series = factory.generate(N2)
    series = np.diff(series, 1)
    factory.reset()
    R, _ = oncd.online_changepoint_detection(
        series,
        partial(oncd.constant_hazard, 250),
        oncd.StudentT(0.1, .01, 1, 0)
    )

    # fig, ax = plt.subplots(figsize=[18, 8])
    # ax = fig.add_subplot(2, 1, 1)
    # ax.plot(series)
    # ax = fig.add_subplot(2, 1, 2, sharex=ax)
    # ax.plot(R[10, 11:-1])
    # plt.show()
    print(change, np.argmax(R[10, 11: -1] > 0.85) + 10)
    return change, np.argmax(R[10, 11: -1] > 0.85) + 10


if __name__ == "__main__":
    S0 = 100
    theta = 0.3
    sig = 0.3
    lmbda = 0.2
    dt = 0.01
    factory = KaratzaFactory(S0, theta, sig, lmbda, dt)

    diff = []
    rev_diff = []
    record = []
    too_early = 0
    start = time.clock()
    for _ in range(500):
        real, est, series = wrapper(factory, 2000)
        record.append([abs(est - real), real, est, series])
        if est > real:
            diff.append(est - real)
        else:
            too_early += 1
            rev_diff.append(real - est)

    print("average diff:", np.mean(rev_diff), "/", np.mean(diff))
    print(too_early)
    print(time.clock() - start)
    record = sorted(record, key=lambda x: -x[0])
    plt.plot(record[0][3])
    plt.axvline(record[0][1], color="r")
    plt.axvline(record[0][2], color="g")
    plt.show()
