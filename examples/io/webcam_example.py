"""
Example: Capture live video from webcam, display with gaze overlay (simulated).
"""

import cv2
import numpy as np
from eyetrace.io import WebcamReader

def main():
    try:
        # Open default webcam, set desired resolution, convert to grayscale
        with WebcamReader(camera_id=0, width=1280, height=720,
                          resize=(640, 360), grayscale=True) as webcam:
            print(f"Webcam opened. FPS estimate: {webcam.fps:.2f}")
            print("Press 'q' to quit.")

            for frame in webcam:
                # Simulate some processing: draw a circle where "gaze" is
                # For demo, just put a moving dot
                h, w = frame.shape[:2]
                center = (w//2 + int(50*np.sin(webcam.frame_count/10)),
                          h//2 + int(30*np.cos(webcam.frame_count/10)))
                # Convert grayscale to BGR for color overlay
                display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.circle(display, center, 10, (0, 255, 0), -1)
                cv2.putText(display, f"Frame: {webcam.frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Webcam - EyeTrace", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except IOError as e:
        print(f"Webcam error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()