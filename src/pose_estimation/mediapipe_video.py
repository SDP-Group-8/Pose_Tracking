from pose_estimation.mediapipe import MediaPipe
from pose_estimation.capture_device import CaptureDevice

class MediaPipeVideo:
    def __init__(self, video):
        self.mp = MediaPipe()
        self.mp.initialize()

        self.video = CaptureDevice(video, False)
     
    def estimate_video(self):
        results = []
        while self.video.is_opened():
            frame_exists, frame = self.video.read()
            if frame_exists:
                timestamp = int(self.video.get_timestamp())
                res = self.mp.process_frame(frame, timestamp)
                results.append(res)
            else:
                break
        self.video.close()
        return results
    