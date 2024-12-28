import cv2
import numpy as np
import os
from nudenet import NudeDetector
def simple_process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(os.path.dirname(input_path), 'simple_processed_' + os.path.basename(input_path))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    NudeDetector.censor('e65d247c162fee1753a60dc25d155ea4.17.jpg')
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simple processing: add frame number to the image
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = NudeDetector().censor(frame)
        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to: {out_path}")

# Usage
input_video_path = "./input_test_Data.mp4"
simple_process_video(input_video_path)