import cv2
import numpy as np
from pathlib import Path

root = Path("G:\Johns Hopkins University\Challenge\davinci_surgical_video")
video_root = Path("C:/Users/DELL1/Videos/4K Video Downloader")

data_root = root / "video_1"
data_root.mkdir(exist_ok=True, parents=True)

sample_root = data_root / "samples"
sample_root.mkdir(exist_ok=True, parents=True)

video_path = video_root / "daVinci Xi Hysterectomy.mp4"
cap = cv2.VideoCapture(str(video_path))

frame_count = 0
skip_count = 0
skip_interval = 200
while cap.isOpened():
    skip_count += 1
    ret, frame = cap.read()
    if frame == None:
        break
    if skip_count < skip_interval:
        continue
    else:
        frame_count += 1
        skip_count = 0
    cv2.imwrite(str(sample_root / "frame_{:05d}.png").format(frame_count), frame)
    print(frame_count)
