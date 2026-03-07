"""
Comprehensive dashboard combining video feed and live plots in a single window.
"""

import cv2
import numpy as np
import time
from typing import Callable, Optional, Dict, Any, List, Literal
from .live_plot import MultiLivePlot


class Dashboard:
    """
    A dashboard that shows the video feed alongside live plots in one window.

    Parameters
    ----------
    video_source : iterable
        An iterator yielding frames (e.g., VideoReader or WebcamReader).
    plot_specs : list of dict
        Specifications for MultiLivePlot. Each dict must contain
        'title', 'ylabel', and optionally 'color'.
    update_interval_ms : int, default 50
        Milliseconds between plot updates (to reduce flicker).
    layout : str, default 'horizontal'
        Arrangement of video and plots: 'horizontal' (side by side) or
        'vertical' (video on top, plots below).
    window_name : str, default 'EyeTrace Dashboard'
        Name of the combined OpenCV window.
    """
    def __init__(self, video_source, plot_specs: List[Dict[str, Any]],
                 update_interval_ms: int = 50,
                 layout: Literal['horizontal', 'vertical'] = 'horizontal',
                 window_name: str = 'EyeTrace Dashboard'):
        self.video = video_source
        self.plot_specs = plot_specs
        self.update_interval = update_interval_ms / 1000.0  # seconds
        self.layout = layout
        self.window_name = window_name
        self.last_plot_update = 0
        self.running = False

        # Create the live plot
        self.plot = MultiLivePlot(plot_specs, maxlen=200)

        # Will store the current plot image
        self.current_plot_img = None

    def _combine_images(self, video_img: np.ndarray, plot_img: np.ndarray) -> np.ndarray:
        """
        Combine video and plot images according to layout.
        Ensures both images have compatible dimensions.
        """
        # Convert plot to BGR if it's RGB (matplotlib produces RGB)
        if plot_img.shape[2] == 3:
            # Assume RGB, convert to BGR for OpenCV
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

        if self.layout == 'horizontal':
            # Resize plot to match video height
            plot_height = video_img.shape[0]
            plot_width = int(plot_img.shape[1] * plot_height / plot_img.shape[0])
            plot_resized = cv2.resize(plot_img, (plot_width, plot_height))
            combined = np.hstack((video_img, plot_resized))
        else:  # vertical
            # Resize plot to match video width
            plot_width = video_img.shape[1]
            plot_height = int(plot_img.shape[0] * plot_width / plot_img.shape[1])
            plot_resized = cv2.resize(plot_img, (plot_width, plot_height))
            combined = np.vstack((video_img, plot_resized))
        return combined

    def _process_and_display(self, process_frame: Callable[[np.ndarray], Dict[str, Any]]):
        """
        Internal loop: reads frames, calls process_frame, updates video and plots.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        for frame in self.video:
            if not self.running:
                break

            # Process frame
            try:
                result = process_frame(frame)
            except Exception as e:
                print(f"Error in process_frame: {e}")
                continue

            disp_frame = result.get('frame', frame)
            timestamp = result.get('timestamp', time.time())
            metrics = result.get('metrics', [])

            # Update plots (throttled)
            if metrics and len(metrics) == len(self.plot_specs):
                if timestamp - self.last_plot_update > self.update_interval:
                    self.plot.update(timestamp, metrics)
                    # Get the current plot as an image
                    self.current_plot_img = self.plot.get_current_figure_as_image()
                    self.last_plot_update = timestamp

            # Combine and display
            if self.current_plot_img is not None:
                combined = self._combine_images(disp_frame, self.current_plot_img)
                cv2.imshow(self.window_name, combined)
            else:
                # Just show video until first plot is ready
                cv2.imshow(self.window_name, disp_frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def run(self, process_frame: Callable[[np.ndarray], Dict[str, Any]]):
        """
        Main loop with exception handling.

        Parameters
        ----------
        process_frame : callable
            Function that takes a frame and returns a dict with keys:
            - 'frame': annotated frame (or None)
            - 'timestamp': float
            - 'metrics': list of float values matching plot_specs order
        """
        self.running = True
        try:
            self._process_and_display(process_frame)
        except KeyboardInterrupt:
            print("Dashboard interrupted by user.")
        except Exception as e:
            print(f"Dashboard error: {e}")
        finally:
            self.close()

    def close(self):
        """Clean up: close plot and destroy windows."""
        self.running = False
        self.plot.close()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()