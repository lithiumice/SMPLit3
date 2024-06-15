import glob
import matplotlib
import cv2
import re
import json


class VideoWrapper:
    video = []

    def __init__(self, filepath):
        self.filepath = filepath
        self.video = cv2.VideoCapture(filepath)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framerate = self.video.get(cv2.CAP_PROP_FPS)

        self.total_sec = self.frame2second(self.total_frames)

    def read(self, seconds=None, frames=None):
        if seconds is not None:
            self.set_current_msec(seconds)
        if frames is not None:
            self.set_current_frame(frames)
        return self.video.read()

    def get_video_reader(self):
        return self.video

    def frame2second(self, frame_no):
        return frame_no / self.framerate

    def second2frame(self, second):
        return int(round(second * self.framerate))

    def set_current_frame(self, cur_frame_no):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_no)

    def set_current_msec(self, cur_msec):
        self.video.set(cv2.CAP_PROP_POS_MSEC, cur_msec)
