"""
Example: Demonstrate geometric calculations: distance, angle, projection.
"""

import numpy as np
from eyetrace.utils.geometry import distance, normalize_vector, angle_between_vectors, project_point_to_line

def main():
    # Points and vectors
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    p3 = np.array([5, 1])
    line_start = np.array([1, 1])
    line_end = np.array([4, 5])

    # Distance
    d = distance(p1, p2)
    print(f"Distance between {p1} and {p2}: {d:.2f}")

    # Normalize vector
    v = p2 - p1
    v_norm = normalize_vector(v)
    print(f"Vector {v} normalized: {v_norm} (length: {np.linalg.norm(v_norm):.2f})")

    # Angle between vectors
    v1 = p2 - p1
    v2 = p3 - p1
    angle = angle_between_vectors(v1, v2, in_degrees=True)
    print(f"Angle between {v1} and {v2}: {angle:.2f} degrees")

    # Project point onto line
    point = np.array([2, 2])
    proj = project_point_to_line(point, line_start, line_end)
    print(f"Projection of {point} onto line ({line_start}->{line_end}): {proj}")
    print(f"Distance from point to line: {distance(point, proj):.2f}")

if __name__ == "__main__":
    main()