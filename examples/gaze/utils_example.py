"""
Example: Utility functions: coordinate conversions and angular velocity.
"""

import numpy as np
from eyetrace.gaze.utils import cartesian_to_spherical, spherical_to_cartesian, angular_velocity

def main():
    # Cartesian to spherical
    x, y, z = 1, 1, 1
    theta, phi, r = cartesian_to_spherical(x, y, z)
    print(f"Cartesian ({x}, {y}, {z}) -> spherical: theta={np.degrees(theta):.1f}°, phi={np.degrees(phi):.1f}°, r={r:.2f}")

    # Spherical to Cartesian
    x2, y2, z2 = spherical_to_cartesian(theta, phi, r)
    print(f"Back to Cartesian: ({x2:.2f}, {y2:.2f}, {z2:.2f})")

    # Angular velocity between two gaze vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    dt = 0.5  # seconds
    omega = angular_velocity(v1, v2, dt)
    print(f"Angular velocity: {np.degrees(omega):.1f}°/s")

if __name__ == "__main__":
    main()