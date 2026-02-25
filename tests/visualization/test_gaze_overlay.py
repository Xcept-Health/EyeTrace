"""
Tests for visualization.gaze_overlay module.
"""

import pytest
import numpy as np
import cv2
from eyetrace.visualization.gaze_overlay import draw_gaze_overlay

def test_draw_gaze_overlay():
    """Test drawing gaze overlay on image."""
    # Créer une image factice
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gaze_point = (320, 240)
    result = draw_gaze_overlay(frame, gaze_point)
    # Vérifie que l'image a la même taille
    assert result.shape == frame.shape
    # Vérifie que des pixels ont changé (le point a été dessiné)
    # On peut vérifier qu'il y a au moins un pixel non nul autour du point
    # Simple : la somme de l'image n'est plus nulle
    assert np.sum(result) > 0