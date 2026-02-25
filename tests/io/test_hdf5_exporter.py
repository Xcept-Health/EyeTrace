"""
Tests for io.video module.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from eyetrace.io.video import VideoReader, WebcamReader

def test_video_reader_from_file():
    """Test VideoReader with a mock video file."""
    with patch('cv2.VideoCapture') as mock_cap:
        instance = mock_cap.return_value
        instance.isOpened.return_value = True
        instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        # Simuler la lecture de frames
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        instance.read.side_effect = [(True, fake_frame)] * 3 + [(False, None)]

        reader = VideoReader('dummy.mp4')
        assert reader.fps == 30
        assert len(reader) == 100
        assert reader.frame_size == (640, 480)

        frames = list(reader)
        assert len(frames) == 3
        instance.read.assert_called()
        reader.release()

def test_webcam_reader():
    """Test WebcamReader with mock."""
    with patch('cv2.VideoCapture') as mock_cap:
        instance = mock_cap.return_value
        instance.isOpened.return_value = True
        instance.get.return_value = 30.0  # fps

        reader = WebcamReader(0, width=1280, height=720, grayscale=True)
        assert reader.fps == 30.0

        # Simuler une seule frame puis fin de flux
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        instance.read.side_effect = [(True, fake_frame), (False, None)]

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 0:  # on ne veut qu'une frame
                break
        assert len(frames) == 1
        assert frames[0].shape == (720, 1280)  # grayscale