from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
from typing import Iterable
import json
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.landmark import Landmark

from pose_estimation.keypoint_encoder import KeypointEncoder
from pose_estimation.keypoint_decoder import KeypointDecoder

@dataclass
class Keypoints:
    left_shoulder: NormalizedLandmark
    right_shoulder: NormalizedLandmark
    left_elbow: NormalizedLandmark
    right_elbow: NormalizedLandmark
    left_wrist: NormalizedLandmark
    right_wrist: NormalizedLandmark

    left_hip: NormalizedLandmark
    right_hip: NormalizedLandmark
    left_knee: NormalizedLandmark
    right_knee: NormalizedLandmark
    left_ankle: NormalizedLandmark
    right_ankle: NormalizedLandmark

    normalized_landmarks: [NormalizedLandmark]

    def to_numpy_positions(self) -> 'np.ndarray':
        return np.asarray([(landmark.x, landmark.y) for landmark in [
                self.left_shoulder,
                self.right_shoulder,
                self.left_elbow,
                self.right_elbow,
                self.left_wrist,
                self.right_wrist,

                self.left_hip,
                self.right_hip,
                self.left_knee,
                self.right_knee,
                self.left_ankle,
                self.right_ankle
            ]
        ])

    def to_dict(self):
        return {
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_elbow": self.left_elbow,
            "right_elbow": self.right_elbow,
            "left_wrist": self.left_wrist,
            "right_wrist": self.right_wrist,

            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_knee": self.left_knee,
            "right_knee": self.right_knee,
            "left_ankle": self.left_ankle,
            "right_ankle": self.right_ankle
        }

    @staticmethod
    def serialize(keypoints: 'Keypoints') -> str:
        return json.dumps(keypoints.to_dict(), cls=KeypointEncoder)

    @staticmethod
    def batch_serialize(iter: Iterable['Keypoints | None']) -> str:
        return "\n".join("None" if keypoints is None else Keypoints.serialize(keypoints) for keypoints in iter)

    @staticmethod
    def deserialize(repr: str) -> 'Keypoints':
        return json.loads(repr, cls=KeypointDecoder)

    @staticmethod
    def batch_deserialize(repr: str) -> Iterable['Keypoints | None']:
        return (None if keypoint_repr == "None" else Keypoints.deserialize(keypoint_repr) for keypoint_repr in repr.split("\n"))

    def to_normalized_landmarks(self) -> [NormalizedLandmark]:
        '''
        Turn object into native mediapipe results object
        :return: mediapipe results object
        '''
        return self.normalized_landmarks

    @classmethod
    def from_normalized_landmarks(cls, normalized_landmarks: [NormalizedLandmark]) -> 'Keypoints':
        '''
        Construct object from detection results
        :param results: native mediapipe results object
        :return: object
        '''
        return cls(
            *[normalized_landmarks[idx] for idx in itertools.chain(range(11, 17), range(23, 29))],
            normalized_landmarks
        )
    
    def from_world_landmarks(cls, world_landmarks: [Landmark]) -> 'Keypoints':
        return cls(
            *[world_landmarks[idx] for idx in itertools.chain(range(11, 17), range(23, 29))],
            world_landmarks
        )

    def get_presences(self, threshold: float) -> np.ndarray[bool]:
        """
        Returns a boolean array of the presence of each keypoint.
        """
        return {k: v.presence > threshold for k, v in self.to_dict().items()}
