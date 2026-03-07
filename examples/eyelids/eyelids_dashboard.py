    """
    Drowsiness monitoring using eyelid metrics.
    Tracks EAR, blink frequency, PERCLOS, and microsleep events via webcam (simulated).
    """

    import cv2
    import numpy as np
    import time
    from collections import deque
    from eyetrace.io import WebcamReader
    from eyetrace.visualization import Dashboard, draw_text_overlay
    from eyetrace.eyelids.blink_detection import detect_blinks, blink_frequency, mean_closure_duration
    from eyetrace.eyelids.perclos import perclos
    from eyetrace.eyelids.microsleep import microsleep_indicator
    from eyetrace.eyelids.ear import both_eyes_ear

    # Simulated EAR generator (replace with real extraction in practice)
    def simulate_ear(t):
        """Simulate EAR with occasional blinks and drowsy trends."""
        base = 0.3
        # Slow oscillation to simulate drowsiness cycles
        drowsy_cycle = 0.05 * np.sin(t / 60)
        # Random blinks (sharp drops)
        if np.random.rand() < 0.02:  # ~2% chance per frame
            blink = -0.15
        else:
            blink = 0.0
        noise = 0.01 * np.random.randn()
        ear = base + drowsy_cycle + blink + noise
        return np.clip(ear, 0.1, 0.5)

    def main():
        # Configuration
        FS = 30  # Hz
        ALERT_EAR_THRESHOLD = 0.25
        ALERT_PERCLOS_THRESHOLD = 20.0  # %
        ALERT_MICROSLEEP_DURATION = 2.0  # seconds

        # Buffers
        buffer_seconds = 60
        buffer_len = int(FS * buffer_seconds)
        ear_buffer = deque(maxlen=buffer_len)
        time_buffer = deque(maxlen=buffer_len)

        # Video source (webcam)
        video = WebcamReader(camera_id=0, width=640, height=480)

        # Plot specifications
        plot_specs = [
            {'title': 'Eye Aspect Ratio (EAR)', 'ylabel': 'ratio', 'color': 'b-'},
            {'title': 'Blink Frequency', 'ylabel': 'blinks/min', 'color': 'g-'},
            {'title': 'PERCLOS (60s)', 'ylabel': '%', 'color': 'r-'}
        ]

        def process_frame(frame):
            t = time.time()

            # Simulate EAR
            ear = simulate_ear(t)

            # Store in buffers
            ear_buffer.append(ear)
            time_buffer.append(t)

            # Compute derived metrics
            blink_freq = 0.0
            perc = 0.0
            microsleep = False
            if len(time_buffer) > 10:
                ear_arr = np.array(ear_buffer)
                times_arr = np.array(time_buffer)

                # Detect blinks
                blinks = detect_blinks(ear_arr, threshold=ALERT_EAR_THRESHOLD)
                duration = times_arr[-1] - times_arr[0]
                if duration > 0:
                    blink_freq = blink_frequency(blinks, duration)

                # PERCLOS over last 60s (or available)
                perc = perclos(ear_arr, threshold=ALERT_EAR_THRESHOLD,
                            window_seconds=60, frame_rate=FS)

                # Microsleep detection
                micro_mask = microsleep_indicator(ear_arr, frame_rate=FS,
                                                ear_threshold=ALERT_EAR_THRESHOLD,
                                                duration_threshold=ALERT_MICROSLEEP_DURATION)
                microsleep = np.any(micro_mask)

            # Annotate frame
            annotated = frame.copy()
            annotated = draw_text_overlay(annotated, [
                f"EAR: {ear:.3f}",
                f"Blinks/min: {blink_freq:.1f}",
                f"PERCLOS: {perc:.1f}%",
                f"Microsleep: {'YES' if microsleep else 'NO'}"
            ], position=(10, 30))

            # Alert if drowsy
            if perc > ALERT_PERCLOS_THRESHOLD or microsleep or ear < ALERT_EAR_THRESHOLD:
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 120), (0, 0, 255), -1)
                annotated = draw_text_overlay(annotated, [
                    " DROWSINESS ALERT "
                ], position=(10, 50), color=(255, 255, 255))

            return {
                'frame': annotated,
                'timestamp': t,
                'metrics': [ear, blink_freq, perc]
            }

        # Create dashboard (video + plots)
        with Dashboard(video, plot_specs, update_interval_ms=100,
                    layout='vertical', window_name='Drowsiness Monitor') as dashboard:
            print("Drowsiness Monitor running. Press 'q' to quit.")
            dashboard.run(process_frame)

    if __name__ == "__main__":
        main()