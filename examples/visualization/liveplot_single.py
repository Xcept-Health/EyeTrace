"""
Example: Display a single metric in real-time using LivePlot.
Simulates a sine wave with noise.
"""

import time
import numpy as np
from eyetrace.visualization import LivePlot

def main():
    # Create a live plot
    with LivePlot(title="Sine Wave", ylabel="Amplitude", maxlen=100, color='g-') as plot:
        t = 0.0
        while True:
            # Simulate data
            value = np.sin(t) + 0.1 * np.random.randn()
            plot.update(t, value)
            t += 0.1
            time.sleep(0.05)  # 20 Hz update

            # Break after 20 seconds for demo (optional)
            if t > 20:
                break

if __name__ == "__main__":
    main()