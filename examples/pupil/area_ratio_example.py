"""
Example: Compute pupil-to-iris area ratio.
"""

from eyetrace.pupil.area_ratio import pupil_iris_area_ratio

def main():
    # Simulated diameters (same unit, e.g., mm)
    pupil_diameter = 4.2
    iris_diameter = 11.8

    ratio = pupil_iris_area_ratio(pupil_diameter, iris_diameter)
    print(f"Pupil diameter: {pupil_diameter} mm")
    print(f"Iris diameter: {iris_diameter} mm")
    print(f"Pupil/Iris area ratio: {ratio:.3f}")

if __name__ == "__main__":
    main()