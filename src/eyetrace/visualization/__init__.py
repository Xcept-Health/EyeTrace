"""
Visualization utilities for EyeTrace.

Provides real-time plotting, gaze overlay on images, and a comprehensive dashboard.
"""

from .live_plot import LivePlot, MultiLivePlot
from .gaze_overlay import draw_gaze_overlay, draw_eye_landmarks, draw_text_overlay
from .dashboard import Dashboard

__all__ = [
    'LivePlot',
    'MultiLivePlot',
    'draw_gaze_overlay',
    'draw_eye_landmarks',
    'draw_text_overlay',
    'Dashboard',
]