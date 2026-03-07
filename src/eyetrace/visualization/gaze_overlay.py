"""
Drawing functions to overlay gaze and eye landmarks on images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union

def draw_gaze_overlay(image: np.ndarray,
                      gaze_point: Optional[Tuple[float, float]] = None,
                      left_pupil: Optional[Tuple[float, float]] = None,
                      right_pupil: Optional[Tuple[float, float]] = None,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2,
                      copy: bool = True) -> np.ndarray:
    """
    Draw gaze point and pupil centers on an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR).
    gaze_point : (x, y) or None
        Estimated gaze point (if available).
    left_pupil, right_pupil : (x, y) or None
        Pupil centers.
    color : tuple
        BGR color for drawings.
    thickness : int
        Line thickness.
    copy : bool, default True
        If True, return a copy; otherwise draw in-place (modify original).

    Returns
    -------
    np.ndarray
        Image with overlays.
    """
    img = image.copy() if copy else image

    if left_pupil is not None:
        cv2.circle(img, tuple(map(int, left_pupil)), 3, color, -1)
    if right_pupil is not None:
        cv2.circle(img, tuple(map(int, right_pupil)), 3, color, -1)

    if gaze_point is not None:
        gx, gy = int(gaze_point[0]), int(gaze_point[1])
        cv2.drawMarker(img, (gx, gy), color, cv2.MARKER_CROSS, 10, thickness)

    return img


def draw_eye_landmarks(image: np.ndarray,
                       left_eye: Optional[np.ndarray] = None,
                       right_eye: Optional[np.ndarray] = None,
                       color_left: Tuple[int, int, int] = (255, 255, 0),
                       color_right: Tuple[int, int, int] = (0, 255, 255),
                       radius: int = 2,
                       copy: bool = True) -> np.ndarray:
    """
    Draw the 6 eye landmarks on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR).
    left_eye, right_eye : np.ndarray, shape (6,2) or None
        Landmark coordinates.
    color_left, color_right : tuple
        BGR colors for left and right eye.
    radius : int
        Radius of the circles.
    copy : bool, default True
        If True, return a copy; otherwise draw in-place.

    Returns
    -------
    np.ndarray
        Image with landmarks.
    """
    img = image.copy() if copy else image
    if left_eye is not None:
        for (x, y) in left_eye.astype(int):
            cv2.circle(img, (x, y), radius, color_left, -1)
    if right_eye is not None:
        for (x, y) in right_eye.astype(int):
            cv2.circle(img, (x, y), radius, color_right, -1)
    return img


def draw_text_overlay(image: np.ndarray,
                      lines: List[str],
                      position: Tuple[int, int] = (10, 30),
                      font_scale: float = 0.6,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2,
                      line_spacing: int = 25,
                      copy: bool = True) -> np.ndarray:
    """
    Draw multiple lines of text on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    lines : list of str
        Text lines.
    position : (x, y)
        Starting position for the first line.
    font_scale : float
        Font scale.
    color : tuple
        BGR color.
    thickness : int
        Thickness of text.
    line_spacing : int
        Vertical spacing between lines.
    copy : bool, default True
        If True, return a copy; otherwise draw in-place.

    Returns
    -------
    np.ndarray
        Image with text.
    """
    img = image.copy() if copy else image
    x, y = position
    for i, line in enumerate(lines):
        y_pos = y + i * line_spacing
        cv2.putText(img, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)
    return img