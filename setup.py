from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


extensions = [
    # Pupil
    Extension(
        "eyetrace.pupil._metrics_cy",
        ["src/eyetrace/pupil/_metrics_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "eyetrace.pupil._dynamics_cy",
        ["src/eyetrace/pupil/_dynamics_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "eyetrace.pupil._area_ratio_cy",
        ["src/eyetrace/pupil/_area_ratio_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    # Eyelids
    Extension(
        "eyetrace.eyelids._ear_cy",
        ["src/eyetrace/eyelids/_ear_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    # Gaze
    Extension(
        "eyetrace.gaze._saccades_cy",
        ["src/eyetrace/gaze/_saccades_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "eyetrace.gaze._fixation_cy",
        ["src/eyetrace/gaze/_fixation_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    # Signal Analysis
    Extension(
        "eyetrace.signal_analysis._entropy_cy",
        ["src/eyetrace/signal_analysis/_entropy_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "eyetrace.signal_analysis._lempel_ziv_cy",
        ["src/eyetrace/signal_analysis/_lempel_ziv_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    name="eyetrace",
    version="0.1.0",
    description="Open-source toolkit for ocular metrics and fatigue detection",
    author="Fildouindé Ariel Shadrac OUEDRAOGO",
    author_email="arielshadrac@gmail.com",
    url="https://github.com/Xcept-Health/EyeTrace.git",
    packages=find_packages(where="src", include=["eyetrace", "eyetrace.*"]),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3", "boundscheck": False},
    ),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)