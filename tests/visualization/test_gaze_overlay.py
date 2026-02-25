"""
Tests for visualization.gaze_overlay module.
"""

import pytest
import numpy as np
import cv2
from eyetrace.visualization.gaze_overlay import draw_gaze_overlay

def test_draw_gaze_overlay():
    """Test drawing gaze overlay on image."""
    # Create a dummy image (black frame)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gaze_point = (320, 240)
    result = draw_gaze_overlay(frame, gaze_point)
    
    # Verify that the output image maintains the same dimensions
    assert result.shape == frame.shape
    
    # Verify that pixels have changed (the point was successfully drawn)
    # Check that the image sum is no longer zero, indicating drawn content
    assert np.sum(result) > 0
