def pupil_iris_area_ratio(double pupil_diameter, double iris_diameter):
    if iris_diameter == 0.0:
        raise ValueError("iris_diameter cannot be zero")
    return (pupil_diameter / iris_diameter) * (pupil_diameter / iris_diameter)