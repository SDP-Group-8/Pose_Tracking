from pose_estimation.mediapipe import MediaPipe
from pose_estimation.capture_device import CaptureDevice
from __future__ import annotations
class MediaPipeVideo:
    def __init__(self, video, parameter_path: str):
        self.mp = MediaPipe()
        self.mp.initialize(parameter_path)

        self.video = CaptureDevice(video, False)
     
    def estimate_video(self):
        results = [None] * self.video.get_frame_count()
        frame_number = -1
        while self.video.is_opened():
            frame_exists, frame = self.video.read()
            timestamp = self.video.get_timestamp()
            frame_number += 1
            if frame_exists and ((not frame_number > 0.0) or timestamp > 0.0):
                res = self.mp.process_frame(frame, int(timestamp))
                if res:
                    results[frame_number] = res
            else:
                break
        self.video.close()
        return results
    