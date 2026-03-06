"""
Example: Read a video file, display basic info, and process frames.

This script uses VideoReader to open a local video, prints properties,
and optionally displays frames (commented out for headless runs).
"""

import cv2
from eyetrace.io import VideoReader

def main(video_path="sample_video.mp4"):
    try:
        with VideoReader(video_path, resize=(640, 480), grayscale=True) as video:
            print(f"Video source: {video.source}")
            print(f"Frame size: {video.frame_size}")
            print(f"FPS: {video.fps:.2f}")
            print(f"Total frames (approx): {len(video)}")
            print("Processing frames...")

            for i, frame in enumerate(video):
                # Example: detect something (here just print progress)
                if i % 30 == 0:
                    print(f"Processed frame {i}/{len(video)}")

                # Uncomment to display frames
                # cv2.imshow('Frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            print(f"Done. Total frames read: {video.frame_count}")

    except IOError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Replace with your video file
    main("path/to/your/video.mp4")