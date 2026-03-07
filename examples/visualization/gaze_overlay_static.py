"""
Example: Load an image, draw gaze point, pupils, and eye landmarks.
Saves the result and displays it.
"""

import cv2
import numpy as np
from eyetrace.visualization import draw_gaze_overlay, draw_eye_landmarks, draw_text_overlay

def main():
    # Create a blank image or load a real one
    # For demo, create a gradient background
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        color = int(100 + 100 * y / 480)
        img[y, :] = [color, color, 150]
    
    # Define fake gaze point and pupils (in pixels)
    gaze_point = (320, 240)
    left_pupil = (250, 200)
    right_pupil = (390, 200)
    
    # Define fake eye landmarks (6 points each) - elliptical shape
    left_landmarks = np.array([
        [230, 180], [250, 170], [270, 180],  # top
        [270, 220], [250, 230], [230, 220]   # bottom
    ])
    
    right_landmarks = np.array([
        [370, 180], [390, 170], [410, 180],
        [410, 220], [390, 230], [370, 220]
    ])
    
    # Draw overlays (copy=False to modify in-place)
    img = draw_gaze_overlay(img, gaze_point, left_pupil, right_pupil,
                           color=(0, 255, 0), thickness=2, copy=False)
    
    img = draw_eye_landmarks(img, left_landmarks, right_landmarks,
                            color_left=(255, 255, 0), color_right=(0, 255, 255),
                            radius=3, copy=False)
    
    img = draw_text_overlay(img, [
        "Gaze point (green cross)",
        "Pupils (green circles)",
        "Eye landmarks (cyan/yellow)"
    ], position=(20, 50), color=(255, 255, 255), line_spacing=30, copy=False)
    
    # Display and save
    cv2.imshow("Gaze Overlay Example", img)
    cv2.imwrite("gaze_overlay_output.jpg", img)
    print("Image saved as gaze_overlay_output.jpg")
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()