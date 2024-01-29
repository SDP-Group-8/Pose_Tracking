import cv2
from datetime import datetime

class CaptureDevice:
    def __init__(self, filepath: str, live: bool = False):
        if filepath.isdigit():
            filepath = int(filepath)
        self.device = cv2.VideoCapture(filepath)
        self.live = live
        self.init_time = datetime.now()

    def get_timestamp(self) -> datetime:
        '''
        Find the timestamp in milliseconds from the start of the video
        '''
        return datetime.now() - self.init_time if self.live else self.device.get(cv2.CAP_PROP_POS_MSEC)

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
