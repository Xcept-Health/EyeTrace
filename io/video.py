"""
Video capture utilities: reading from files or webcam.
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple, Union

class VideoReader:
    """
    Read frames from a video file.

    Parameters
    ----------
    path : str
        Path to the video file.
    resize : tuple, optional
        If given, (width, height) to resize each frame.
    grayscale : bool, default False
        If True, convert frames to grayscale.
    """
    def __init__(self, path: str, resize: Optional[Tuple[int, int]] = None,
                 grayscale: bool = False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        self.resize = resize
        self.grayscale = grayscale
        self._frame_count = 0

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over frames (yield until end).
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self._process_frame(frame)
            self._frame_count += 1
            yield frame

    def __len__(self) -> int:
        """
        Return total number of frames (if known).
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_count(self) -> int:
        """Number of frames read so far."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Original (width, height) of video frames."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply resize and grayscale conversion."""
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def release(self):
        """Release the video capture."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamReader(VideoReader):
    """
    Read frames from a webcam.

    Parameters
    ----------
    camera_id : int, default 0
        ID of the camera (0 for default).
    width : int, optional
        Desired width (if supported by camera).
    height : int, optional
        Desired height.
    resize : tuple, optional
        If given, (width, height) to resize each frame.
    grayscale : bool, default False
        If True, convert frames to grayscale.
    """
    def __init__(self, camera_id: int = 0, width: Optional[int] = None,
                 height: Optional[int] = None,
                 resize: Optional[Tuple[int, int]] = None,
                 grayscale: bool = False):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {camera_id}")
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.resize = resize
        self.grayscale = grayscale
        self._frame_count = 0

    # __len__ is not meaningful for live camera
    def __len__(self):
        raise NotImplementedError("Webcam has no predetermined length")

    @property
    def fps(self) -> float:
        """Estimated frames per second (may be approximate)."""
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback