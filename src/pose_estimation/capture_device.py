from __future__ import annotations
import cv2
import time
from typing import Tuple

class CaptureDevice:
    def __init__(self, filepath: str, live: bool = False, dimensions: Tuple = None):
        if filepath.isdigit():
            filepath = int(filepath)
        self.device = cv2.VideoCapture(filepath)
        self.live = live
        self.init_time = time.perf_counter_ns()
        self.dimensions = dimensions

    def get_timestamp(self) -> float:
        '''
        Find the timestamp in milliseconds from the start of the video
        '''
        if self.live:
            return ((time.perf_counter_ns() - self.init_time) * 1e-6)
        else:
            return self.device.get(cv2.CAP_PROP_POS_MSEC)

    def is_opened(self) -> bool:
        '''
        Determine whether the stream could be opened
        '''
        return self.device.isOpened()

    def read(self):
        '''
        Read the next frame from the input device
        '''
        exists, frame = self.device.read()
        return exists, cv2.resize(frame, self.dimensions) if self.dimensions else frame
    
    def get_width(self) -> int:
        '''
        Get width of the video stream in pixels
        '''
        return int(self.device.get(3))
    
    def get_height(self) -> int:
        '''
        Get height of the video stream in pixels
        '''
        return int(self.device.get(4))
    
    def get_frame_count(self) -> int:
        return int(self.device.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        self.device.release()
