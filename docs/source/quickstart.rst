Quick Start
===========

This page provides simple examples to get you started with EyeTrace.

Compute Eye Aspect Ratio (EAR) from a webcam
--------------------------------------------

.. code-block:: python

   import cv2
   import mediapipe as mp
   from eyetrace.eyelids import eye_aspect_ratio
   from eyetrace.eyelids.utils import extract_both_eyes

   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

   cap = cv2.VideoCapture(0)
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       results = face_mesh.process(rgb)
       if results.multi_face_landmarks:
           h, w, _ = frame.shape
           left, right = extract_both_eyes(results.multi_face_landmarks[0], w, h)
           ear_left = eye_aspect_ratio(left)
           ear_right = eye_aspect_ratio(right)
           ear_avg = (ear_left + ear_right) / 2.0
           cv2.putText(frame, f"EAR: {ear_avg:.3f}", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       cv2.imshow('EyeTrace - EAR', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   cap.release()
   cv2.destroyAllWindows()

Extract pupil diameter from a video file
-----------------------------------------

.. code-block:: python

   from eyetrace.pupil import extract_pupil_diameter
   from eyetrace.io import VideoReader
   from eyetrace.utils.landmarks import extract_iris_landmarks_from_mediapipe
   import mediapipe as mp
   import cv2

   mp_face_mesh = mp.solutions.face_mesh
   with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
       with VideoReader("subject_01.mp4") as video:
           for frame in video:
               rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               results = face_mesh.process(rgb)
               if results.multi_face_landmarks:
                   h, w, _ = frame.shape
                   left_iris = extract_iris_landmarks_from_mediapipe(
                       results.multi_face_landmarks[0], w, h, eye='left')
                   right_iris = extract_iris_landmarks_from_mediapipe(
                       results.multi_face_landmarks[0], w, h, eye='right')
                   diam_left = extract_pupil_diameter(left_iris, w, h)
                   diam_right = extract_pupil_diameter(right_iris, w, h)
                   diam = (diam_left + diam_right) / 2.0
                   print(f"Frame {video.frame_count}: pupil diameter = {diam:.2f} px")

More examples
-------------

Check the `examples/ <https://github.com/Xcept-Health/EyeTrace/tree/main/examples>`_ folder on GitHub for more complete scripts, including:

- Clinical pupillary light reflex test
- Parkinson's disease screening using micro-saccades
- Drowsiness detection with PERCLOS and blink frequency
- Real-time dashboard with multiple metrics