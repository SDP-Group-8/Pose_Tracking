import json
from typing import Iterable

from pose_estimation.keypoints import Keypoints
from pose_estimation.keypoint_encoder import KeypointEncoder
from pose_estimation.keypoint_decoder import KeypointDecoder

class KeypointSerializer:
    @staticmethod
    def serialize(keypoints: 'Keypoints') -> str:
        return json.dumps(keypoints.to_dict(), cls=KeypointEncoder)

    @staticmethod
    def batch_serialize(iter: Iterable['Keypoints | None']) -> str:
        return "\n".join("None" if keypoints is None else KeypointSerializer.serialize(keypoints) for keypoints in iter)
    
    @staticmethod
    def deserialize(repr: str) -> 'Keypoints':
        return json.loads(repr, cls=KeypointDecoder)

    @staticmethod
    def batch_deserialize(repr: str) -> Iterable['Keypoints | None']:
        return (None if keypoint_repr == "None" else KeypointSerializer.deserialize(keypoint_repr) for keypoint_repr in repr.split("\n"))