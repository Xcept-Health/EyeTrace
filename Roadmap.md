# Roadmap

**EyeTrace** aims to become the most comprehensive open‑source toolkit for ocular metric extraction and fatigue detection. This roadmap outlines the development plan, from core foundations to advanced features and distribution. We welcome contributors at every stage!

---

## Overview

The project is organized into five major modules, each containing multiple metrics:

| Module                  | Metrics                                                                                     |
|-------------------------|---------------------------------------------------------------------------------------------|
| **Pupil Dynamics**      | Variance, standard deviation, coefficient of variation (CV), normalized diameter (NPD), constriction/dilation speeds, hippus amplitude, Z‑score, first derivative, pupil/iris area ratio |
| **Eyelids & Blinks**    | Eye Aspect Ratio (EAR), PERCLOS, blink frequency, mean closure duration (MCD), eyelid closing/opening speeds, long‑blink ratio, eyelid symmetry, EAR jerk, micro‑sleep indicator |
| **Gaze Movements**      | Saccade velocity/acceleration, gaze entropy (Shannon), fixation duration/dispersion, gaze centroid, saccade/fixation ratio, 3D gaze vector, vergence speed, pupil eccentricity |
| **Head Pose & Face**    | Pitch/roll/yaw, head angular velocity, Mouth Aspect Ratio (MAR), yawning frequency, nose stability, neck flexion angle, inter‑pupillary distance (IPD), postural sag |
| **Advanced Signal Analysis** | FFT, power ratios (LF/HF), sample entropy, Hurst exponent, Lempel‑Ziv complexity, Higuchi fractal dimension, mutual information (L/R synchronization), SNR, trend slope, **Karlinska Sleepiness Score (KSS)** prediction model |

All metrics will be implemented in **pure Python** first, then optimized with **Cython** for performance-critical parts. The project will be distributed as **pre‑compiled wheels** for Linux, macOS, and Windows.

---

## Phase 1: Foundations (Months 1–2)

**Goal:** Set up the project structure, core dependencies, and implement the first two modules (Pupil Dynamics and Eyelids) with basic functionality.

### Tasks
- [ ] Initialize repository with the structure described in `CONTRIBUTING.md`.
- [ ] Set up `setup.py`, `pyproject.toml`, and `requirements.txt` with core dependencies:
  - `numpy`, `scipy`, `opencv-python`, `mediapipe`, `pandas`, `matplotlib`, `cython`, `pytest`
- [ ] Implement **pupil detection** using MediaPipe iris landmarks.
  - Extract pupil diameter and center.
  - Provide a pure Python fallback.
- [ ] Implement **basic pupil metrics**:
  - Variance, standard deviation, coefficient of variation (CV).
  - Normalized pupil diameter (NPD).
  - First derivative of diameter.
- [ ] Implement **EAR (Eye Aspect Ratio)** calculation.
  - Pure Python version.
  - Cython version for speed.
  - Conditional import with fallback.
- [ ] Implement **blink detection** using EAR thresholding.
  - Detect blink events, compute blink frequency.
- [ ] Write **unit tests** for all implemented functions.
- [ ] Create **basic examples**:
  - Real‑time EAR display from webcam.
  - Pupil diameter extraction from a video file.
- [ ] Set up **continuous integration** (GitHub Actions) to run tests on multiple Python versions (3.8–3.12).

**Deliverables:**
- Working `pupil` and `eyelids` modules with the above metrics.
- Example scripts in `examples/`.
- Initial documentation in `README.md` and docstrings.

---

## Phase 2: Enrichment of Core Modules (Months 3–4)

**Goal:** Complete the remaining metrics in Pupil Dynamics and Eyelids, and begin Gaze and Head Pose modules.

### Tasks
#### Pupil Dynamics (completion)
- [ ] Constriction speed (light response).
- [ ] Dilation speed (recovery).
- [ ] Hippus amplitude.
- [ ] Z‑score normalization.
- [ ] Pupil/iris area ratio.

#### Eyelids (completion)
- [ ] PERCLOS (percentage of eyelid closure over time).
- [ ] Mean closure duration (MCD).
- [ ] Eyelid closing speed.
- [ ] Eyelid reopening speed.
- [ ] Long‑blink ratio (blinks > threshold duration).
- [ ] Eyelid symmetry (left vs. right eye EAR correlation).
- [ ] EAR jerk (rate of change of EAR).
- [ ] Micro‑sleep indicator (boolean if EAR < threshold for T seconds).

#### Gaze Module (start)
- [ ] Set up gaze direction estimation from eye landmarks.
- [ ] Implement saccade detection (velocity‑based).
- [ ] Saccade velocity and acceleration.
- [ ] Gaze entropy (Shannon) over fixation periods.

#### Head Pose Module (start)
- [ ] Estimate head pose (pitch, roll, yaw) using MediaPipe face mesh and solvePnP.
- [ ] Compute head angular velocity.
- [ ] Mouth Aspect Ratio (MAR) and yawning detection.
- [ ] Nose stability (variance of nose position).

#### Cross‑cutting
- [ ] Add Cython optimizations for all new performance‑sensitive functions.
- [ ] Expand unit tests to cover new features.
- [ ] Write examples for each new metric.

**Deliverables:**
- Complete `pupil` and `eyelids` modules with all listed metrics.
- Initial `gaze` and `head_pose` modules with basic functions.
- Updated documentation.

---

## Phase 3: Advanced Gaze, Head Pose, and Signal Analysis (Months 5–7)

**Goal:** Finish Gaze and Head Pose modules, and implement all advanced signal analysis metrics.

### Tasks
#### Gaze Module (completion)
- [ ] Fixation duration and dispersion.
- [ ] Gaze centroid (average gaze point over time).
- [ ] Saccade/fixation ratio.
- [ ] 3D gaze vector.
- [ ] Vergence speed (for binocular tracking).
- [ ] Pupil eccentricity (angle correction).

#### Head Pose Module (completion)
- [ ] Neck flexion angle (if multiple body landmarks available).
- [ ] Inter‑pupillary distance (IPD).
- [ ] Postural sag (change in eye height over time).

#### Advanced Signal Analysis
- [ ] Fast Fourier Transform (FFT) of pupil or gaze signals.
- [ ] Power ratios (LF/HF) from FFT.
- [ ] Sample entropy.
- [ ] Hurst exponent (memory analysis).
- [ ] Lempel‑Ziv complexity (pattern repetitiveness).
- [ ] Higuchi fractal dimension (signal roughness).
- [ ] Mutual information (left/right eye synchronization).
- [ ] Signal‑to‑noise ratio (tracking quality).
- [ ] Trend slope (temporal degradation).

#### Integration
- [ ] Combine selected metrics into a **Karlinska Sleepiness Score (KSS) prediction model** (simple regression or classifier).
- [ ] Provide a high‑level function that returns a fatigue estimate.

#### Cython & Performance
- [ ] Optimize all signal analysis functions with Cython (where beneficial).
- [ ] Profile and benchmark to ensure real‑time capability.

**Deliverables:**
- Complete `gaze` and `head_pose` modules.
- Full `signal_analysis` module.
- KSS prediction model (basic version).
- Comprehensive test suite and examples.

---

## Phase 4: Packaging, Distribution, and Documentation (Month 8)

**Goal:** Prepare EyeTrace for a wide audience by creating professional documentation, building wheels, and publishing on PyPI.

### Tasks
- [ ] Write **full documentation** with Sphinx, hosted on ReadTheDocs.
  - API reference for all modules.
  - Tutorials and user guides.
  - Explanation of each metric and its interpretation.
- [ ] Set up **cibuildwheel** in GitHub Actions to generate wheels for:
  - Linux (manylinux x86_64, aarch64)
  - macOS (universal2: x86_64 + arm64)
  - Windows (64-bit)
- [ ] Test wheels on clean environments (Docker, VMs).
- [ ] Publish **first release** (v0.1.0) on TestPyPI, then on PyPI.
- [ ] Update `README.md` with installation instructions, badges, and links.
- [ ] Create a **project website** (optional) or GitHub Pages site with examples.

**Deliverables:**
- EyeTrace available via `pip install eyetrace`.
- Pre‑compiled wheels for all major platforms.
- Professional documentation.

---

## Phase 5: Community Building and Long‑term Maintenance (Ongoing)

**Goal:** Grow the contributor base, respond to user feedback, and continuously improve.

### Tasks
- [ ] Encourage contributions via “good first issues”.
- [ ] Regularly review and merge pull requests.
- [ ] Add more examples and use cases.
- [ ] Explore performance enhancements (GPU acceleration with CUDA/OpenCL).
- [ ] Integrate with other biometric sensors (EEG, ECG) if requested.
- [ ] Plan for version 1.0 with stable API.

---

## How You Can Help

We welcome contributions of all kinds:
- **Code** – implement a metric, optimize with Cython, fix bugs.
- **Documentation** – improve tutorials, translate, write docstrings.
- **Testing** – test on different hardware/OS, report issues.
- **Ideas** – suggest new features or improvements.

Check the [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started.

---


src/
└── eyetrace/
    ├── pupil/                  # Module I : Dynamique pupillaire
    │   ├── __init__.py         # Expose l'API publique, import conditionnel
    │   ├── features.py         # Implémentations Python pures (variance, std, CV, NPD, etc.)
    │   ├── _features_cy.pyx    # Optimisations Cython pour les fonctions critiques
    │   ├── constriction.py     # Détection des phases de constriction/dilatation (Python)
    │   ├── _constriction_cy.pyx# Version Cython des algorithmes de détection
    │   ├── hippus.py           # Calcul de l'amplitude de l'hippus (Python)
    │   ├── _hippus_cy.pyx      # Version Cython (filtrage, FFT)
    │   ├── zscore.py           # Normalisation Z‑Score (Python)
    │   ├── _zscore_cy.pyx      # Optionnel (peut rester en Python si simple)
    │   ├── derivative.py       # Dérivée première (Python)
    │   ├── _derivative_cy.pyx  # Version Cython pour calcul en temps réel
    │   ├── area_ratio.py       # Ratio surface pupille/iris (Python)
    │   └── _area_ratio_cy.pyx  # Optimisation Cython (calcul de rayons)
    │
    ├── eyelids/                 # Module II : Paupières et clignements
    │   ├── __init__.py
    │   ├── ear.py               # EAR (Python pur) – déjà créé
    │   ├── _ear_cy.pyx          # EAR (Cython)
    │   ├── perclos.py           # PERCLOS (Python)
    │   ├── _perclos_cy.pyx      # PERCLOS (Cython)
    │   ├── blink_detection.py   # Détection des clignements, fréquence, MCD (Python)
    │   ├── _blink_detection_cy.pyx # Optimisation Cython
    │   ├── eyelid_speed.py      # Vitesses de fermeture/réouverture (Python)
    │   ├── _eyelid_speed_cy.pyx # Cython
    │   ├── long_blink.py        # Ratio clignements longs (Python)
    │   ├── symmetry.py          # Symétrie oculaire (Python)
    │   ├── jerk.py              # Taux de changement de l'EAR (Python)
    │   ├── microsleep.py        # Indicateur de micro‑sommeil (Python)
    │   └── _microsleep_cy.pyx   # Cython (si nécessaire)
    │
    ├── gaze/                     # Module III : Mouvements du regard
    │   ├── __init__.py
    │   ├── saccades.py           # Détection, vitesse, accélération (Python)
    │   ├── _saccades_cy.pyx      # Cython (boucles de détection)
    │   ├── fixation.py           # Durée, dispersion, centroïde (Python)
    │   ├── _fixation_cy.pyx      # Cython
    │   ├── entropy.py            # Entropie de Shannon du regard (Python)
    │   ├── _entropy_cy.pyx       # Cython (calcul de distribution)
    │   ├── ratio.py              # Ratio saccade/fixation (Python)
    │   ├── vector_3d.py          # Vecteur de regard 3D (Python)
    │   ├── vergence.py           # Vitesse de vergence (Python)
    │   ├── _vergence_cy.pyx      # Cython
    │   ├── eccentricity.py       # Excentricité pupillaire (Python)
    │   └── _eccentricity_cy.pyx  # Cython
    │
    ├── head_pose/                 # Module IV : Posture et visage
    │   ├── __init__.py
    │   ├── angles.py              # Pitch, roll, yaw (Python, avec solvePnP)
    │   ├── _angles_cy.pyx         # Cython (si optimisations possibles)
    │   ├── angular_velocity.py    # Vitesse angulaire (Python)
    │   ├── mar.py                 # Mouth Aspect Ratio (Python)
    │   ├── _mar_cy.pyx            # Cython
    │   ├── yawning.py             # Fréquence de bâillement (Python)
    │   ├── nose_stability.py      # Stabilité du nez (Python)
    │   ├── neck_angle.py          # Angle de flexion du cou (Python)
    │   ├── ipd.py                 # Distance inter‑pupillaire (Python)
    │   ├── _ipd_cy.pyx            # Cython
    │   ├── postural_sag.py        # Affaissement postural (Python)
    │   └── _postural_sag_cy.pyx   # Cython
    │
    ├── signal_analysis/            # Module V : Analyse avancée
    │   ├── __init__.py
    │   ├── fourier.py              # FFT, ratios de puissance (Python, utilise scipy/numpy)
    │   ├── _fourier_cy.pyx         # Éventuellement des parties personnalisées
    │   ├── entropy.py              # Sample Entropy (Python)
    │   ├── _entropy_cy.pyx         # Cython (boucles critiques)
    │   ├── hurst.py                # Exposant de Hurst (Python)
    │   ├── _hurst_cy.pyx           # Cython
    │   ├── lempel_ziv.py           # Complexité LZ (Python)
    │   ├── _lempel_ziv_cy.pyx      # Cython
    │   ├── higuchi.py              # Dimension fractale de Higuchi (Python)
    │   ├── _higuchi_cy.pyx         # Cython
    │   ├── mutual_info.py          # Information mutuelle G/D (Python)
    │   ├── _mutual_info_cy.pyx     # Cython
    │   ├── snr.py                  # Rapport signal/bruit (Python)
    │   ├── trend.py                # Pente de dérive (Python)
    │   ├── kss.py                  # Modèle KSS (Python)
    │   └── _kss_cy.pyx             # Optionnel (si le modèle est simple, pas nécessaire)
    │
    ├── io/                          # Entrées/sorties
    │   ├── __init__.py
    │   ├── video.py                 # Lecture vidéo, webcam
    │   ├── csv_exporter.py          # Export des métriques au format CSV
    │   └── hdf5_exporter.py         # Export HDF5 (optionnel)
    │
    ├── visualization/               # Visualisation
    │   ├── __init__.py
    │   ├── live_plot.py             # Affichage temps réel des métriques
    │   ├── gaze_overlay.py          # Superposition du regard sur l'image
    │   └── dashboard.py             # Tableau de bord complet
    │
    └── utils/                        # Fonctions utilitaires transversales
        ├── __init__.py
        ├── landmarks.py              # Extraction des landmarks MediaPipe
        ├── filtering.py              # Filtres (Kalman, moving average)
        ├── geometry.py               # Calculs géométriques (angles, distances)
        └── math_helpers.py           # Fonctions mathématiques diverses

*This roadmap is a living document and may evolve based on community feedback and technological advances. Last updated: February 2026.*