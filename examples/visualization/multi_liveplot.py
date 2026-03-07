"""
Example: Display multiple metrics in real-time using MultiLivePlot.
Simulates pupil diameter, EAR, and blink rate.
"""

import time
import numpy as np
from eyetrace.visualization import MultiLivePlot

def main():
    # Define plot specifications for three metrics
    specs = [
        {'title': 'Pupil Diameter', 'ylabel': 'mm', 'color': 'b-'},
        {'title': 'Eye Aspect Ratio (EAR)', 'ylabel': 'ratio', 'color': 'g-'},
        {'title': 'Blink Events', 'ylabel': 'binary', 'color': 'r-'}
    ]

    # Create multi-plot with 200 points history
    with MultiLivePlot(specs, maxlen=200, sharex=True) as plot:
        t = 0.0
        print("MultiLivePlot running. Close the window to exit.")
        
        while True:
            # Simulate realistic metrics
            pupil = 4.0 + 0.5 * np.sin(t / 2) + 0.1 * np.random.randn()
            ear = 0.3 + 0.1 * np.sin(t) + 0.02 * np.random.randn()
            ear = np.clip(ear, 0.2, 0.5)
            blink = 1.0 if np.random.rand() < 0.02 else 0.0  # 2% chance of blink
            
            # Update plot
            plot.update(t, [pupil, ear, blink])
            
            t += 0.1
            time.sleep(0.05)  # 20 Hz update

if __name__ == "__main__":
    main()