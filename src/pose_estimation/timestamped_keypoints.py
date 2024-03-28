from dataclasses import dataclass
import time
from pose_estimation.keypoints import Keypoints

@dataclass
class TimestampedKeypoints:
    timestamp: time
    keypoints: Keypoints

