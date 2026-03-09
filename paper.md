---
title: 'EyeTrace: A Python library for real-time eye-tracking analysis and clinical biomarker extraction'
tags:
  - Python
  - eye-tracking
  - pupillometry
  - neurology
  - Parkinson's disease
  - fatigue detection
  - open-source
authors:
  - name: Fidlouindé Ariel Shadrac OUEDRAOGO
    orcid: 0009-0003-3419-5985
    affiliation: "1, 2"
affiliations:
  - name: Independent Researcher, Burkina Faso
    index: 1
  - name: xcept-health, Burkina Faso
    index: 2
date: 2026-03-09
bibliography: paper.bib
---

# Summary

EyeTrace is an open-source Python library for comprehensive analysis of eye-tracking
data, with a focus on real-time processing and clinical applications. It provides
modular tools for extracting and analysing features from video-based eye trackers,
including pupil dynamics (diameter, hippus, pupillary light reflex), gaze behaviour
(fixations, saccades, vergence), eyelid movements (blink rate, PERCLOS, microsleeps),
and head pose estimation. The library integrates with MediaPipe [@mediapipe] for
facial landmark detection and offers real-time visualisation dashboards built with
OpenCV [@opencv]. EyeTrace aims to lower the barrier for researchers and clinicians
to develop biomarkers for neurological disorders such as Parkinson's disease, fatigue
monitoring, and autonomic dysfunction.

# Statement of Need

Eye-tracking is a powerful non-invasive technique for studying cognitive and
neurological processes. However, existing open-source tools often focus on offline
analysis or require proprietary hardware, making them inaccessible in
resource-limited settings. In sub-Saharan Africa, where access to proprietary
neurological diagnostic equipment is restricted, clinicians lack affordable and
flexible alternatives for the early detection of neurological disorders such as
Parkinson's disease and fatigue-related conditions.

EyeTrace addresses this gap by providing:

- A unified interface for real-time feature extraction from standard webcam video
  feeds, removing the need for specialised hardware.
- A modular design enabling easy integration with existing research pipelines.
- Implementation of validated algorithms for pupillary dynamics (hippus amplitude,
  constriction speed), gaze analysis (fixation dispersion, saccade detection), and
  eyelid metrics (blink duration, PERCLOS [@dinges1998perclos]).
- Built-in visualisation tools for live monitoring, suitable for clinical feedback
  and experimental use.
- Practical examples simulating clinical scenarios (Parkinson's screening, fatigue
  detection) to facilitate adoption by non-specialist users.

By making these tools freely available, EyeTrace supports research in neurology,
sleep medicine, human factors, and psychology, with particular relevance to
low-resource clinical environments.

# Usage Example

The following example demonstrates a real-time dashboard for monitoring pupillary
dynamics using a standard webcam:

```python
import time
import cv2
from eyetrace.io import WebcamReader
from eyetrace.visualization import Dashboard
from eyetrace.pupil.core import extract_pupil_diameter
from eyetrace.pupil.dynamics import hippus_amplitude
from eyetrace.pupil.metrics import coefficient_variation

diam_buffer = []

def process_frame(frame):
    # Extract iris landmarks via MediaPipe (internal pipeline)
    left_diam, right_diam = extract_pupil_diameter(frame)
    diam = (left_diam + right_diam) / 2.0
    diam_buffer.append(diam)

    metrics = {}
    if len(diam_buffer) >= 30:
        metrics["hippus"] = hippus_amplitude(diam_buffer[-90:], fs=30)
        metrics["cv"] = coefficient_variation(diam_buffer[-90:])

    annotated = frame.copy()
    cv2.putText(annotated, f"Diameter: {diam:.2f} mm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if metrics:
        cv2.putText(annotated, f"Hippus: {metrics['hippus']:.3f} mm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"CV: {metrics['cv']:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return {
        "frame": annotated,
        "timestamp": time.time(),
        "metrics": [diam, metrics.get("hippus", 0), metrics.get("cv", 0)],
    }

plot_specs = [
    {"title": "Pupil Diameter", "ylabel": "mm"},
    {"title": "Hippus Amplitude", "ylabel": "mm"},
    {"title": "Coefficient of Variation", "ylabel": "%"},
]

with Dashboard(WebcamReader(camera_id=0), plot_specs) as dashboard:
    dashboard.run(process_frame)
```

This example streams live video, extracts pupil diameter frame by frame, computes
real-time hippus amplitude and coefficient of variation, and displays both the
annotated video feed and live metric plots in a single window.

# Mathematics

EyeTrace implements several validated mathematical models for clinical feature
extraction.

The **Eye Aspect Ratio (EAR)** [@soukupova2016real] is used for blink detection:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2\,\|p_1 - p_4\|}$$

where $p_1$ to $p_6$ are the six eye landmark coordinates. A blink is detected
when EAR falls below a calibrated threshold.

**Pupillary hippus amplitude** is estimated via bandpass filtering (0.5–4 Hz)
followed by root mean square (RMS) calculation:

$$\text{Hippus} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x_{\text{filt}}(n)^2}$$

where $x_{\text{filt}}$ is the bandpass-filtered pupil diameter signal and $N$ is
the number of samples in the analysis window.

**PERCLOS** (Percentage of Eye Closure) [@dinges1998perclos] is defined as the
proportion of time the eye is 80% or more closed over a given epoch:

$$\text{PERCLOS} = \frac{\text{frames with EAR} < 0.2}{\text{total frames}} \times 100$$

# Acknowledgements

The author thanks the developers of MediaPipe, OpenCV, NumPy, and the broader
scientific Python ecosystem. This work was motivated by the need for accessible
neurological screening tools in resource-limited clinical environments in
sub-Saharan Africa.

# References