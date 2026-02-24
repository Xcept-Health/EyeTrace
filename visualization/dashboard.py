"""
Comprehensive dashboard combining video feed and live plots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Callable, Optional, Dict, Any
import time


class Dashboard:
    """
    A dashboard that shows the video feed alongside live plots.

    Parameters
    ----------
    video_source : VideoReader or WebcamReader
        Source of frames.
    plot_specs : list of dict
        Specifications for MultiLivePlot.
    update_interval_ms : int, default 50
        Milliseconds between plot updates (to reduce flicker).
    """
    def __init__(self, video_source, plot_specs: list, update_interval_ms: int = 50):
        self.video = video_source
        self.plot_specs = plot_specs
        self.update_interval = update_interval_ms / 1000.0  # seconds
        self.last_plot_update = 0

        # Initialize plot (matplotlib figure)
        self.fig, self.axes = plt.subplots(len(plot_specs), 1, figsize=(6, 8))
        if len(plot_specs) == 1:
            self.axes = [self.axes]
        self.lines = []
        for i, spec in enumerate(plot_specs):
            line, = self.axes[i].plot([], [], spec.get('color', 'b-'))
            self.lines.append(line)
            self.axes[i].set_title(spec['title'])
            self.axes[i].set_ylabel(spec['ylabel'])
            self.axes[i].grid(True)
        self.axes[-1].set_xlabel('Time (s)')
        self.fig.tight_layout()

        # Convert matplotlib figure to OpenCV image once
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout(pad=2)

        self.data_buffers = [{'x': [], 'y': []} for _ in plot_specs]

    def update_plots(self, x: float, y_values: list):
        """Add new data points and refresh the figure."""
        for i, val in enumerate(y_values):
            self.data_buffers[i]['x'].append(x)
            self.data_buffers[i]['y'].append(val)
            # Keep last 200 points
            if len(self.data_buffers[i]['x']) > 200:
                self.data_buffers[i]['x'] = self.data_buffers[i]['x'][-200:]
                self.data_buffers[i]['y'] = self.data_buffers[i]['y'][-200:]

        # Update line data
        for i in range(len(self.plot_specs)):
            self.lines[i].set_data(self.data_buffers[i]['x'], self.data_buffers[i]['y'])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        # Redraw figure
        self.canvas.draw()
        # Convert to numpy array
        buf = self.canvas.buffer_rgba()
        plot_img = np.asarray(buf)
        # Convert RGBA to BGR for OpenCV
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        return plot_img

    def run(self, process_frame: Callable[[np.ndarray], Dict[str, Any]]):
        """
        Main loop: read frames, process, update video and plots.

        Parameters
        ----------
        process_frame : callable
            Function that takes a frame and returns a dict with keys:
            - 'frame': annotated frame (or None)
            - 'timestamp': float
            - 'metrics': list of float values matching plot_specs order
        """
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Plots', cv2.WINDOW_NORMAL)

        for frame in self.video:
            start_time = time.time()

            # Process frame
            result = process_frame(frame)
            disp_frame = result.get('frame', frame)
            timestamp = result.get('timestamp', time.time())
            metrics = result.get('metrics', [])

            # Show video
            cv2.imshow('Video', disp_frame)

            # Update plots (throttled)
            if timestamp - self.last_plot_update > self.update_interval:
                if metrics and len(metrics) == len(self.plot_specs):
                    plot_img = self.update_plots(timestamp, metrics)
                    cv2.imshow('Plots', plot_img)
                    self.last_plot_update = timestamp

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()