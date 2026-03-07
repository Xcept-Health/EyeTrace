"""
Real-time plotting of metrics using matplotlib (in non-blocking mode) with context manager support.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import List, Optional, Dict, Any
import weakref
import cv2


class LivePlot:
    """
    A live-updating line plot for a single metric with context manager.

    Parameters
    ----------
    title : str
        Plot title.
    ylabel : str
        Label for y-axis.
    maxlen : int, default 100
        Maximum number of points to display.
    color : str, default 'b-'
        Matplotlib line style and color.
    xlabel : str, default 'Time (s)'
        Label for x-axis.
    """
    def __init__(self, title: str, ylabel: str, maxlen: int = 100,
                 color: str = 'b-', xlabel: str = 'Time (s)'):
        self.maxlen = maxlen
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.color = color
        self.x_data = deque(maxlen=maxlen)
        self.y_data = deque(maxlen=maxlen)

        # Create figure and axes
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], color)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)

        # Turn on interactive mode
        plt.ion()
        self.fig.show()

        # Ensure figure is closed when object is deleted
        self._finalizer = weakref.finalize(self, self._close_figure, self.fig)

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
        """Close the plot window explicitly."""
        self._finalizer()

    @staticmethod
    def _close_figure(fig):
        """Static method to close figure."""
        plt.close(fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MultiLivePlot:
    """
    Live plot with multiple subplots for different metrics, with context manager.

    Parameters
    ----------
    specs : list of dict
        Each dict must contain 'title', 'ylabel', and optionally 'color' and 'xlabel'.
        Example: [{'title': 'Pupil Size', 'ylabel': 'mm', 'color': 'r-'}]
    maxlen : int, default 100
        Maximum number of points per line.
    sharex : bool, default True
        Whether subplots share the x-axis.
    """
    def __init__(self, specs: List[Dict[str, Any]], maxlen: int = 100, sharex: bool = True):
        self.maxlen = maxlen
        self.n_plots = len(specs)
        self.data = [deque(maxlen=maxlen) for _ in range(self.n_plots)]
        self.times = deque(maxlen=maxlen)

        # Create subplots
        self.fig, self.axes = plt.subplots(self.n_plots, 1, sharex=sharex)
        if self.n_plots == 1:
            self.axes = [self.axes]
        self.lines = []
        for i, spec in enumerate(specs):
            line, = self.axes[i].plot([], [], spec.get('color', 'b-'))
            self.lines.append(line)
            self.axes[i].set_title(spec['title'])
            self.axes[i].set_ylabel(spec['ylabel'])
            self.axes[i].grid(True)
            if 'xlabel' in spec and i == self.n_plots-1:
                self.axes[i].set_xlabel(spec['xlabel'])
        if not sharex:
            self.axes[-1].set_xlabel(specs[-1].get('xlabel', 'Time (s)'))
        else:
            self.axes[-1].set_xlabel('Time (s)')
        self.fig.tight_layout()

        plt.ion()
        self.fig.show()

        self._finalizer = weakref.finalize(self, self._close_figure, self.fig)

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
        if len(y_values) != self.n_plots:
            raise ValueError(f"Expected {self.n_plots} values, got {len(y_values)}")
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
        self._finalizer()
        
    def get_current_figure_as_image(self) -> np.ndarray:
        """
        Convert the current matplotlib figure to an RGB numpy array.
        Useful for embedding in OpenCV windows.
        """
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        # Convert RGBA to RGB (ignore alpha)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    @staticmethod
    def _close_figure(fig):
        plt.close(fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()