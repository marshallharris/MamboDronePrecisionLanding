#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    screenArea = 640 * 480
    distanceAndAreaData = np.array([
        [20, 144601/screenArea],
        [40, 50885/screenArea],
        [60, 21330/screenArea],
        [80, 12390/screenArea],
        [100, 7994/screenArea],
        [140, 4279/screenArea],
        [160, 3242/screenArea],
        [200, 2214/screenArea],
    ])

    fig, ax = plt.subplots()
    ax.set(xlabel="Distance cm", ylabel='target area px', title='Area as a function of distance')
    ax.plot(distanceAndAreaData[:, 0], distanceAndAreaData[:, 1])
    ax.grid()

    plt.show()

