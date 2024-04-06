#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description="Filename of csv data to read")
    parser.add_argument("filename")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    df = pd.read_csv(args.filename)

    timestamps = df['timestamp']
    timestamps = timestamps - timestamps.min()

    x_error = df['x_error']
    y_error = df['y_error']

    fig, ax = plt.subplots(2)
    ax[0].plot(timestamps, x_error, 'o')
    ax[1].plot(timestamps, y_error, 'o')

    plt.show()

