"""
Video capture utilities: reading from files, webcam, or network streams.
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple, Union, Any
import re

class VideoReader:
    """
    Read frames from a video file or network stream.

    Parameters
    ----------
    source : str or int
        Path to video file, camera index (int), or URL (e.g., 'rtsp://...').
    resize : tuple, optional
        If given, (width, height) to resize each frame.
    grayscale : bool, default False
        If True, convert frames to grayscale.
    **kwargs
        Additional parameters passed to cv2.VideoCapture (e.g., backend).
    """
    def __init__(self, source: Union[str, int],
                 resize: Optional[Tuple[int, int]] = None,
                 grayscale: bool = False,
                 **kwargs):
        self.source = source
        self.resize = resize
        self.grayscale = grayscale
        self.kwargs = kwargs
        self.cap = None
        self._frame_count = 0
        self._open_capture()

    def _open_capture(self):
        """Open the video capture with the given source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
        # Apply any additional properties set in kwargs
        for prop, value in self.kwargs.items():
            if hasattr(cv2, prop):
                self.cap.set(getattr(cv2, prop), value)

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
        Return total number of frames (if known). For live sources, raises error.
        """
        if self.is_live:
            raise NotImplementedError("Live sources have no predetermined length")
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def is_live(self) -> bool:
        """Return True if the source is a live camera or stream."""
        return isinstance(self.source, int) or self.source.startswith(('rtsp://', 'http://'))

    @property
    def frame_count(self) -> int:
        """Number of frames read so far."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """
        Frames per second of the video.
        For live sources, may return an estimate or a default value.
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Try to get from metadata or fallback
            if self.is_live:
                fps = 30.0  # Common fallback for webcams
            else:
                fps = 30.0  # arbitrary fallback
        return fps

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
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamReader(VideoReader):
    """
    Convenience class for webcam capture (camera index).
    """
    def __init__(self, camera_id: int = 0,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 resize: Optional[Tuple[int, int]] = None,
                 grayscale: bool = False):
        kwargs = {}
        if width is not None:
            kwargs['CAP_PROP_FRAME_WIDTH'] = width
        if height is not None:
            kwargs['CAP_PROP_FRAME_HEIGHT'] = height
        super().__init__(camera_id, resize=resize, grayscale=grayscale, **kwargs)