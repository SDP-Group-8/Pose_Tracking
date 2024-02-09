from pose_estimation.mediapipe import MediaPipe
from pose_estimation.capture_device import CaptureDevice

class MediaPipeVideo:
    def __init__(self, video):
        self.mp = MediaPipe()
        self.mp.initialize()

        self.video = CaptureDevice(video, False)
     
    def estimate_video(self):
        results = [None] * self.video.get_frame_count()
        frame_number = -1
        while self.video.is_opened():
            frame_exists, frame = self.video.read()
            frame_number += 1
            if frame_exists:
                timestamp = int(self.video.get_timestamp())
                res = self.mp.process_frame(frame, timestamp)
                if res:
                    results[frame_number] = res
            else:
                break
        self.video.close()
        return results
    