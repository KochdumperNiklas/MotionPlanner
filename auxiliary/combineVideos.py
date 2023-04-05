import cv2
from os import listdir
import os
from os.path import isfile, join

# get all available video files
folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = join(folder, 'videos')
videos = [f for f in listdir(path) if isfile(join(path, f)) and not f.startswith('video')]

# create the new video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(join(path, "video.mp4"), fourcc, 10, (2500, 1500))

# write all the frames sequentially to the new video
for v in videos:
    curr_v = cv2.VideoCapture(join(path, v))
    while curr_v.isOpened():
        # get return value and curr frame of curr video
        r, frame = curr_v.read()
        if not r:
            break
        # write the frame
        video.write(frame)

# Save the video
video.release()