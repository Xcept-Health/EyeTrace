"""
Example: Read an RTSP stream from an IP camera.

Note: The camera must support RTSP and you need the correct URL.
"""

from eyetrace.io import VideoReader

def main(rtsp_url="rtsp://username:password@192.168.1.100:554/stream"):
    try:
        with VideoReader(rtsp_url, resize=(640, 480)) as stream:
            print(f"Connected to {rtsp_url}")
            print(f"Frame size: {stream.frame_size}, FPS: {stream.fps:.2f}")
            print("Reading stream...")

            for i, frame in enumerate(stream):
                # Process frame (here just count)
                if i % 100 == 0:
                    print(f"Received {i} frames")
                # Break after 500 frames for demo
                if i >= 500:
                    break

            print(f"Stream ended. Total frames read: {stream.frame_count}")

    except IOError as e:
        print(f"Failed to open stream: {e}")

if __name__ == "__main__":
    # Replace with your actual RTSP URL
    main("rtsp://your_camera_ip/stream")