"""
Tests for io.video module.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from eyetrace.io.video import VideoCapture, VideoWriter, list_cameras

def test_video_capture_initialization():
    """Test VideoCapture initialization."""
    # On peut mocker cv2.VideoCapture pour éviter la webcam réelle
    with patch('cv2.VideoCapture') as mock_cap:
        instance = mock_cap.return_value
        instance.isOpened.return_value = True
        cap = VideoCapture(0)
        assert cap.is_opened() is True
        mock_cap.assert_called_with(0)

def test_video_capture_read():
    """Test reading a frame."""
    with patch('cv2.VideoCapture') as mock_cap:
        instance = mock_cap.return_value
        instance.isOpened.return_value = True
        # Simuler une frame
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        instance.read.return_value = (True, fake_frame)
        cap = VideoCapture(0)
        ret, frame = cap.read()
        assert ret is True
        assert frame.shape == (480, 640, 3)

def test_video_writer():
    """Test VideoWriter initialization and write."""
    with patch('cv2.VideoWriter') as mock_writer:
        instance = mock_writer.return_value
        instance.isOpened.return_value = True
        writer = VideoWriter('output.avi', 30, (640, 480))
        assert writer.is_opened() is True
        # Test write
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
        instance.write.assert_called_with(frame)

def test_list_cameras():
    """Test listing available cameras (mock)."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Simuler que les indices 0 et 1 fonctionnent
        def side_effect(index):
            instance = MagicMock()
            if index in [0, 1]:
                instance.isOpened.return_value = True
            else:
                instance.isOpened.return_value = False
            return instance
        mock_cap.side_effect = side_effect
        cameras = list_cameras(max_tested=3)
        assert cameras == [0, 1]