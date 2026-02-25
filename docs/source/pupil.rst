Pupil Module
============

The ``eyetrace.pupil`` module provides tools for extracting and analyzing pupil dynamics from video frames. It includes functions for pupil diameter estimation, variability metrics, constriction/dilation speeds, hippus, and pupillary light reflex (PLR) analysis.

.. automodule:: eyetrace.pupil
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

.. toctree::
   :maxdepth: 2

   pupil.core
   pupil.metrics
   pupil.dynamics
   pupil.area_ratio
   pupil.plr

Core Functions
--------------

The core module handles pupil and iris landmark extraction and diameter calculation.

.. automodule:: eyetrace.pupil.core
   :members:
   :undoc-members:
   :show-inheritance:

Basic Statistical Metrics
-------------------------

Functions for computing variance, standard deviation, coefficient of variation, normalized diameter, and Z‑scores.

.. automodule:: eyetrace.pupil.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Metrics
---------------

Analysis of pupil changes over time: first derivative, constriction/dilation speeds, hippus amplitude.

.. automodule:: eyetrace.pupil.dynamics
   :members:
   :undoc-members:
   :show-inheritance:

Pupil-to-Iris Area Ratio
------------------------

Simple calculation of the ratio of pupil area to iris area.

.. automodule:: eyetrace.pupil.area_ratio
   :members:
   :undoc-members:
   :show-inheritance:

Pupillary Light Reflex (PLR)
-----------------------------

Complete analysis of the pupillary response to a light stimulus, including latency, amplitude, velocities, and recovery time.

.. automodule:: eyetrace.pupil.plr
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Here is a complete example of using the pupil module to process a video file and compute various metrics:

.. code-block:: python

   import cv2
   import mediapipe as mp
   import numpy as np
   from eyetrace.pupil import (
       extract_pupil_diameter,
       variance,
       std_dev,
       coefficient_variation,
       first_derivative,
       constriction_speed,
       dilation_speed,
       hippus_amplitude,
       pupil_iris_area_ratio,
       plr_analysis
   )
   from eyetrace.utils.landmarks import extract_iris_landmarks_from_mediapipe

   mp_face_mesh = mp.solutions.face_mesh
   with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
       cap = cv2.VideoCapture(0)  # or use VideoReader
       diameters = []
       times = []
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           results = face_mesh.process(rgb)
           if results.multi_face_landmarks:
               h, w = frame.shape[:2]
               left_iris = extract_iris_landmarks_from_mediapipe(
                   results.multi_face_landmarks[0], w, h, 'left')
               right_iris = extract_iris_landmarks_from_mediapipe(
                   results.multi_face_landmarks[0], w, h, 'right')
               diam_left = extract_pupil_diameter(left_iris, w, h)
               diam_right = extract_pupil_diameter(right_iris, w, h)
               diam = (diam_left + diam_right) / 2.0
               diameters.append(diam)
               times.append(cv2.getTickCount() / cv2.getTickFrequency())
           # Break after 30 seconds or any condition

       diameters = np.array(diameters)
       times = np.array(times) - times[0]

       print(f"Variance: {variance(diameters):.3f}")
       print(f"Std dev: {std_dev(diameters):.3f}")
       print(f"CV: {coefficient_variation(diameters):.2f}%")
       print(f"First derivative (mean): {np.mean(first_derivative(diameters, times)):.3f}")
       print(f"Hippus amplitude: {hippus_amplitude(diameters, fs=30):.3f} px")

       # If a light stimulus was applied at t=stimulus_time
       stimulus_time = 5.0
       plr = plr_analysis(diameters, times, stimulus_time)
       print(f"PLR latency: {plr['latency']:.3f} s")
       print(f"PLR amplitude: {plr['amplitude_percent']:.1f}%")