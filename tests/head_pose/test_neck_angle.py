"""
Tests for head_pose.neck_angle module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.neck_angle import neck_flexion_angle

def test_neck_flexion_angle():
    """Test neck flexion angle (placeholder)."""
    # Pour l'instant, la fonction retourne 0.0
    angle = neck_flexion_angle(None)
    assert angle == 0.0