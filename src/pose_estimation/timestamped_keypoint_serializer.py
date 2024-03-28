import json
from typing import Iterable

from pose_estimation.timestamped_keypoints import TimestampedKeypoints
from pose_estimation.timestamped_keypoints_decoder import TimestampedKeypointsDecoder
from pose_estimation.keypoint_encoder import KeypointEncoder

class TimestampedKeypointsSerializer:
    @staticmethod
    def serialize(keypoints: 'TimestampedKeypoints'):
        return json.dumps({
            TimestampedKeypointsDecoder.timestamp_field_name: keypoints.timestamp,
            TimestampedKeypointsDecoder.keypoints_field_name: keypoints.keypoints.to_dict()
        }, cls=KeypointEncoder)

    @staticmethod
    def batch_serialize(iter: Iterable['Keypoints | None']):
        return "\n".join("None" if keypoints is None else TimestampedKeypointsSerializer.serialize(keypoints) for keypoints in iter)

    @staticmethod
    def deserialize(repr: str):
        return json.loads(repr, cls=TimestampedKeypointsDecoder)

    @staticmethod
    def batch_deserialize(repr: str):
        return (None if keypoint_repr == "None" else TimestampedKeypointsSerializer.deserialize(keypoint_repr) for keypoint_repr in repr.split("\n"))
