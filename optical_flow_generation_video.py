# Python 2/3 compatibility
from __future__ import print_function

from pathlib import Path
import numpy as np
import cv2
# import video


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':

    root = Path("G:\Johns Hopkins University\Challenge\davinci_surgical_video")
    video_root = Path("C:/Users/DELL1/Videos/4K Video Downloader")

    data_root = root / "video_1"
    data_root.mkdir(exist_ok=True, parents=True)

    video_path = video_root / "daVinci Xi Hysterectomy.mp4"
    cap = cv2.VideoCapture(str(video_path))

    # # cam = video.create_capture(fn)
    # # ret, prev = cam.read()
    # # data_root = Path("G:\Johns Hopkins University\Challenge\miccai_challenge_2018_training_data")
    # # seq_index = 1
    # # image_filename_list = list((data_root / ('seq_' + str(seq_index)) / 'left_frames').glob('frame*'))
    #
    # # prev = cv2.imread( str(image_filename_list[0]) )
    # # prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # frame_count = 0
    # skip_count = 0
    # skip_interval = 5
    # while cap.isOpened():
    #     skip_count += 1
    #     ret, frame = cap.read()
    #     # if skip_count < skip_interval:
    #     #     continue
    #     # else:
    #     #     frame_count += 1
    #     #     skip_count = 0
    #     # cv2.imwrite(str(data_root / "frame_{:05d}.png").format(frame_count), frame)
    #     print(frame_count)

    ret, frame = cap.read()
    prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = True
    cur_glitch = frame.copy()

    count = 0
    skip_interval = 5
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if count < skip_interval:
            continue
        else:
            count = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print(gray.shape, flow.shape)
        prevgray = gray

        cv2.imshow('flow', draw_flow(gray, flow))
    #     if show_hsv:
    #         cv2.imshow('flow HSV', draw_hsv(flow))
    #     if show_glitch:
    #         cv2.imshow("current", cur_glitch)
    #         cur_glitch = warp_flow(cur_glitch, flow)
    #         cv2.imshow('glitch', cur_glitch)
    #         cv2.imshow("next", frame)
    #
    #         cur_glitch = frame.copy()
    #
    #     # if show_glitch:
    #     #     cv2.imshow("prev", cur_glitch)
    #     #     img_warped = warp_flow(img, flow).copy()
    #     #     cv2.imshow('glitch', img_warped)
    #
        ch = cv2.waitKey()
    #     # if ch == 27:
    #     #     break
    #     # if ch == ord('1'):
    #     #     show_hsv = not show_hsv
    #     #     print('HSV flow visualization is', ['off', 'on'][show_hsv])
    #     # if show_glitch:
    #     #     cur_glitch = img.copy()
    #     #     print('glitch is', ['off', 'on'][show_glitch])
    # cv2.destroyAllWindows()
