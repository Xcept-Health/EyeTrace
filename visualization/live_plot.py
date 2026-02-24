"""
Real-time plotting of metrics using matplotlib (in non-blocking mode).
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import List, Optional, Dict, Any


class LivePlot:
    """
    A simple live-updating line plot for a single metric.

    Parameters
    ----------
    title : str
        Plot title.
    ylabel : str
        Label for y-axis.
    maxlen : int, default 100
        Maximum number of points to display.
    """
    def __init__(self, title: str, ylabel: str, maxlen: int = 100):
        self.maxlen = maxlen
        self.x_data = deque(maxlen=maxlen)
        self.y_data = deque(maxlen=maxlen)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_title(title)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self.fig.show()

    def update(self, x: float, y: float):
        """
        Add a new point and refresh the plot.

        Parameters
        ----------
        x : float
            x-coordinate (e.g., timestamp).
        y : float
            y-coordinate (metric value).
        """
        self.x_data.append(x)
        self.y_data.append(y)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)


class MultiLivePlot:
    """
    Live plot with multiple subplots for different metrics.

    Parameters
    ----------
    specs : list of dict
        Each dict must contain 'title', 'ylabel', and optionally 'color'.
    maxlen : int, default 100
        Maximum number of points per line.
    """
    def __init__(self, specs: List[Dict[str, Any]], maxlen: int = 100):
        self.maxlen = maxlen
        self.n_plots = len(specs)
        self.data = [deque(maxlen=maxlen) for _ in range(self.n_plots)]
        self.times = deque(maxlen=maxlen)

        plt.ion()
        self.fig, self.axes = plt.subplots(self.n_plots, 1, sharex=True)
        if self.n_plots == 1:
            self.axes = [self.axes]
        self.lines = []
        for i, spec in enumerate(specs):
            line, = self.axes[i].plot([], [], spec.get('color', 'b-'))
            self.lines.append(line)
            self.axes[i].set_title(spec['title'])
            self.axes[i].set_ylabel(spec['ylabel'])
            self.axes[i].grid(True)
        self.axes[-1].set_xlabel('Time (s)')
        self.fig.tight_layout()
        self.fig.show()

    def update(self, x: float, y_values: List[float]):
        """
        Add a new point for all metrics.

        Parameters
        ----------
        x : float
            Timestamp.
        y_values : list of float
            Values for each subplot (must match length of specs).
        """
        self.times.append(x)
        for i, val in enumerate(y_values):
            self.data[i].append(val)
            self.lines[i].set_data(self.times, self.data[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)