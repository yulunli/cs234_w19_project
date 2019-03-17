import numpy as np
import matplotlib.pyplot as plt

'''
Plot performance graphs of regret and average number of incorrect
dosing decisions vs. number of training examples seen over N training
runs. Each data point is averaged over N runs with error bars for min and max

x: numpy array of 0:delta:5000, length 5000/delta + 1
regrets: list or numpy array of shape [N, len(x)]
fractions: list or numpy array of shape [N, len(x)]
'''
def plot(x, regrets, fractions):
    x = np.array(x)
    regrets = np.array(regrets)
    fractions = np.array(fractions)

    plt.figure(1)
    regret_mean = np.mean(regrets, axis=0)
    regret_err = np.stack((regret_mean - np.min(regrets, axis=0), np.max(regrets, axis=0) - regret_mean))
    plt.errorbar(x, regret_mean, yerr=regret_err)

    plt.figure(2)
    fraction_mean = np.mean(fractions, axis=0)
    fraction_err = np.stack((fraction_mean - np.min(fractions, axis=0), np.max(fractions, axis=0) - fraction_mean))
    plt.errorbar(x, fraction_mean, yerr=fraction_err)
    plt.show()
