# EyeTrace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Xcept-Health/EyeTrace/ci.yml?branch=main)](https://github.com/Xcept-Health/EyeTrace/actions)
[![Documentation Status](https://readthedocs.org/projects/eyetrace/badge/?version=latest)](https://eyetrace.readthedocs.io/en/latest/?badge=latest)
[![status](https://joss.theoj.org/papers/f0fceaea27a58675ed4e59499b9be103/status.svg)](https://joss.theoj.org/papers/f0fceaea27a58675ed4e59499b9be103)

> **The brain's state is written in your pupils.**  
> EyeTrace is an open-source toolkit that decodes this language — extracting over 50 physiological and behavioral metrics from facial videos to assess fatigue, drowsiness, and cognitive load.

---

## Table of Contents
- [Why EyeTrace?](#-why-eyetrace)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Modules](#-modules)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Tests](#-tests)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## Why EyeTrace?

The pupil is a window to the brain. It reacts to cognitive load, fatigue, surprise, and even subconscious processes. By tracking pupil dynamics, eyelid movements, and gaze patterns, we can objectively measure:

- **Drowsiness** – before it becomes dangerous (for drivers, pilots, machine operators)
- **Mental workload** – in human‑factors research
- **Neurological disorders** – such as Parkinson’s or Alzheimer’s (altered pupil response)
- **Emotional arousal** – in psychology experiments

EyeTrace brings together state‑of‑the‑art computer vision and signal processing to make these measurements accessible to researchers, developers, and enthusiasts — all in one well‑documented, modular Python package.

---

## Features

| Category               | Metrics                                                                                     |
|------------------------|---------------------------------------------------------------------------------------------|
| **Pupil Dynamics**     | Diameter, variance, coefficient of variation (CV), normalized diameter (NPD), constriction/dilation speeds, hippus amplitude, Z‑score, first derivative, pupil/iris area ratio |
| **Eyelid & Blink**     | Eye Aspect Ratio (EAR), PERCLOS, blink frequency, mean closure duration (MCD), eyelid speed (closing/opening), long‑blink ratio, eyelid symmetry, EAR jerk, microsleep indicator |
| **Gaze Movements**     | Saccade velocity/acceleration, gaze entropy (Shannon), fixation duration/dispersion, gaze centroid, saccade/fixation ratio, 3D gaze vector, vergence speed, pupil eccentricity |
| **Head Pose & Face**   | Pitch/roll/yaw, head angular velocity, Mouth Aspect Ratio (MAR), yawning frequency, nose stability, neck flexion angle, inter‑pupillary distance (IPD), postural sag |
| **Advanced Signal Analysis** | FFT, power ratios (LF/HF), sample entropy, Hurst exponent, Lempel‑Ziv complexity, Higuchi fractal dimension, mutual information (L/R synchronization), SNR, trend slope, **Karlinska Sleepiness Score (KSS)** prediction model |

---

## Installation

### From PyPI
```bash
pip install eyetrace
```

### From source (latest development version)
```bash
git clone https://github.com/Xcept-Health/EyeTrace.git
cd EyeTrace
pip install -e .
```

**Dependencies** are automatically installed: `numpy`, `scipy`, `opencv-python`, `mediapipe`, `pandas`, `matplotlib`.  
For GPU acceleration (MediaPipe), follow the [official instructions](https://google.github.io/mediapipe/getting_started/python.html).

---

## Quick Start

### Compute EAR from a webcam stream
```python
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
```

### Extract pupil diameter from a video file
```python
from eyetrace.pupil import extract_pupil_diameter
from eyetrace.io import VideoReader
from eyetrace.utils.landmarks import extract_iris_landmarks_from_mediapipe
import mediapipe as mp

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
```

More examples are available in the [`examples/`](examples/) folder.

---

## Modules

- `eyetrace.pupil` – Pupil detection, diameter, constriction/dilation, variability metrics.
- `eyetrace.eyelids` – EAR, blink detection, PERCLOS, microsleeps, eyelid speed.
- `eyetrace.gaze` – Saccade/fixation classification, gaze entropy, vergence.
- `eyetrace.head_pose` – Head orientation (PnP), mouth opening, yawning detection.
- `eyetrace.signal_analysis` – Time‑series analysis: FFT, entropy, fractal dimension, fusion models.
- `eyetrace.io` – Video/camera input, data export (CSV, HDF5).
- `eyetrace.utils` – Landmark extraction, filtering, geometry, and math helpers.
- `eyetrace.visualization` – Real‑time plotting, gaze overlay, dashboard.

All performance‑critical functions are accelerated with Cython, while pure Python fallbacks ensure compatibility on any system.

---

## Documentation

Full API reference, tutorials, and background theory are available at **[eyetrace.readthedocs.io](https://eyetrace.readthedocs.io)** (coming soon).

- [Getting Started](https://eyetrace.readthedocs.io/en/latest/getting_started.html)
- [Feature Descriptions](https://eyetrace.readthedocs.io/en/latest/features.html)
- [Contribution Guide](https://eyetrace.readthedocs.io/en/latest/contributing.html)

---

## Running Tests

```bash
pytest tests/ -v
```

We use `pytest` and `pytest-cov`. To generate a coverage report:
```bash
pytest tests/ --cov=eyetrace --cov-report=html
```

---

## Contributing

Contributions are **welcome** and **appreciated**!  
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

Areas where we need help:
- Implementing new metrics (see the list above)
- Optimizing performance (Cython, numba, GPU)
- Writing documentation and examples
- Testing on diverse datasets

---

## License

EyeTrace is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for the full text.

---

## Citation

If you use EyeTrace in your research, please cite:

```bibtex
@software{eyetrace2026,
  author = {Xcept Health},
  title = {EyeTrace: Open-source toolkit for ocular metrics and fatigue detection},
  year = {2026},
  url = {https://github.com/Xcept-Health/EyeTrace},
  note = {The brain's state is written in your pupils.}
}
```

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev) for real‑time face and iris tracking.
- [Soukupová & Čech](http://www.inf.ufrgs.br/~rppesquisa/FutGen/TCCs/2016-2/Diogo_Maldaner%20-%20artigofinal.pdf) for the EAR formulation.
- The open‑source community for countless invaluable tools.
- All **contributors** and **early adopters** who help shape EyeTrace.

---

**EyeTrace** – Because the eyes don't lie.  

