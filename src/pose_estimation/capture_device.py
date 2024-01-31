import cv2
import time

class CaptureDevice:
    def __init__(self, filepath: str, live: bool = False):
        if filepath.isdigit():
            filepath = int(filepath)
        self.device = cv2.VideoCapture(filepath)
        self.live = live
        self.init_time = time.perf_counter_ns()

    def get_timestamp(self) -> float:
        '''
        Find the timestamp in milliseconds from the start of the video
        '''
        if self.live:
            return ((time.perf_counter_ns() - self.init_time) * 1e-3)
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
        return self.device.read()
    
    def get_width(self) -> int:
        '''
        Get width of the video stream in pixels
        '''
        return self.device.get(3)
    
    def get_height(self) -> int:
        '''
        Get height of the video stream in pixels
        '''
        return self.device.get(4)
    
    def close(self):
        self.device.release()
