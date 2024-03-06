from dataclasses import dataclass
import time
import json
from typing import Iterable

from pose_estimation.timestamped_keypoints_decoder import TimestampedKeypointsDecoder
from pose_estimation.keypoints import Keypoints

@dataclass
class TimestampedKeypoints:
    timestamp: time
    keypoints: Keypoints

    @staticmethod
    def serialize(keypoints: 'TimestampedKeypoints'):
        return json.dumps({
            TimestampedKeypointsDecoder.timestamp_field_name: keypoints.timestamp,
            TimestampedKeypointsDecoder.keypoints_field_name: keypoints.keypoints.to_dict()
        })
    
    @staticmethod
    def batch_serialize(iter: Iterable['Keypoints | None']):
        return "\n".join("None" if keypoints is None else TimestampedKeypoints.serialize(keypoints) for keypoints in iter)

    @staticmethod
    def deserialize(repr: str):
        return json.loads(repr, cls=TimestampedKeypointsDecoder)

    @staticmethod
    def batch_deserialize(repr: str):
        return (None if keypoint_repr == "None" else TimestampedKeypoints.deserialize(keypoint_repr) for keypoint_repr in repr.split("\n"))
