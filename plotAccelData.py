#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description="Filename of csv data to read")
    parser.add_argument("filename")
    return parser.parse_args()

if __name__ == "__main__":
    # Read the CSV file
    args = parseArgs()
    df = pd.read_csv(args.filename)

    # Create a colormap
    colormap = plt.cm.viridis

    timestamps = df['timestamp']
    # Normalize timestamp to range [0, 1]
    normalize = plt.Normalize(vmin=timestamps.min(), vmax=timestamps.max())
    colors = colormap(normalize(timestamps))

    # Extract x, y, and z coordinates
    x = df['pos_x']
    y = df['pos_y']
    z = -1 * df['pos_z']
 
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    scatter = ax.scatter(x, y, z, c=colors, marker='o')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Position Plot')

        # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Timestamp')

    # Show plot
    plt.show()