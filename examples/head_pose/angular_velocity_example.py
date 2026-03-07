"""
Example: Compute head angular velocity from a time series of angles.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.head_pose.angular_velocity import head_angular_velocity

def main():
    # Simulate head movement: a slow turn to the right then back
    fs = 30  # Hz
    t = np.arange(0, 10, 1/fs)

    # Yaw: start 0°, go to 30°, back to 0°
    yaw = 15 * (1 - np.cos(2 * np.pi * 0.2 * t))  # max 30°
    pitch = 5 * np.sin(2 * np.pi * 0.1 * t)       # small oscillation
    roll = np.zeros_like(t)

    # Convert to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Compute angular velocity
    pitch_vel, roll_vel, yaw_vel = head_angular_velocity(pitch_rad, roll_rad, yaw_rad, t)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(t, yaw, label='Yaw (deg)')
    plt.plot(t, pitch, label='Pitch (deg)')
    plt.legend()
    plt.ylabel('Angle (deg)')
    plt.title('Head Angles')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, np.degrees(yaw_vel), label='Yaw velocity (deg/s)')
    plt.plot(t, np.degrees(pitch_vel), label='Pitch velocity (deg/s)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (deg/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()