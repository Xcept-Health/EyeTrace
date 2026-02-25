.. _eyelids-module:

Eyelids Module
==============

.. module:: eyetrace.eyelids
   :synopsis: Eyelid and blink analysis.

The `eyelids` module provides functions and classes to analyze eyelid movements, detect blinks, compute PERCLOS, measure eyelid speeds, assess symmetry, and detect microsleep events.

**Main features:**

- Eye Aspect Ratio (EAR) calculation
- Blink detection and statistics (frequency, duration, long blink ratio)
- PERCLOS (percentage of eyelid closure)
- Eyelid closing and opening speeds
- Eyelid symmetry (correlation between left and right EAR)
- EAR jerk (rate of change of eyelid movement)
- Microsleep detection (prolonged eye closure)

.. rubric:: Submodules

.. toctree::
   :maxdepth: 1

   eyelids.ear
   eyelids.blink_detection
   eyelids.perclos
   eyelids.eyelid_speed
   eyelids.symmetry
   eyelids.jerk
   eyelids.microsleep

.. rubric:: Usage Example

.. code-block:: python

   import cv2
   import mediapipe as mp
   from eyetrace.eyelids import eye_aspect_ratio
   from eyetrace.eyelids.utils import extract_both_eyes

   mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
   cap = cv2.VideoCapture(0)

   while True:
       ret, frame = cap.read()
       if not ret:
           break
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       results = mp_face_mesh.process(rgb)
       if results.multi_face_landmarks:
           h, w, _ = frame.shape
           left, right = extract_both_eyes(results.multi_face_landmarks[0], w, h)
           ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
           cv2.putText(frame, f"EAR: {ear:.2f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
       cv2.imshow('EAR Demo', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   cap.release()
   cv2.destroyAllWindows()

.. rubric:: Module Contents

.. automodule:: eyetrace.eyelids
   :members:
   :undoc-members:
   :show-inheritance: