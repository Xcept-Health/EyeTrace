"""
Example: Estimate Karlinska Sleepiness Score from dummy features.
"""

from eyetrace.signal_analysis.kss import karlinska_sleepiness_score

def main():
    # Simulated features from an eye-tracking session
    features = {
        'perclos': 15.2,           # % of eye closure
        'blink_frequency': 12.5,    # blinks per minute
        'pupil_variance': 0.08,     # variance of pupil diameter
        'head_movement': 0.02       # variance of head angle
    }

    kss = karlinska_sleepiness_score(features)
    print(f"Estimated KSS: {kss:.1f} (1=alert, 9=sleepy)")

    # Try with sleepy features
    sleepy_features = {
        'perclos': 45.0,
        'blink_frequency': 25.0,
        'pupil_variance': 0.25,
        'head_movement': 0.15
    }
    kss_sleepy = karlinska_sleepiness_score(sleepy_features)
    print(f"Estimated KSS (sleepy scenario): {kss_sleepy:.1f}")

if __name__ == "__main__":
    main()