.. _io.video:

Video Input/Output
==================

.. automodule:: eyetrace.io.video
   :members:
   :undoc-members:
   :show-inheritance:

This submodule contains classes for reading frames from video files and
webcams:

- :class:`VideoReader` : read frames from a video file with optional resizing
  and grayscale conversion.
- :class:`WebcamReader` : capture live video from a camera, inheriting from
  `VideoReader` and allowing frame size configuration.

Both classes support iteration over frames and context manager usage.