from dataclasses import dataclass
import itertools
import numpy as np
from typing import Iterable
import json
from __future__ import annotations
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

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

    normalized_landmarks: list[NormalizedLandmark]

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

    def to_normalized_landmarks(self) -> list[NormalizedLandmark]:
        '''
        Turn object into native mediapipe results object
        :return: mediapipe results object
        '''
        return self.normalized_landmarks

    @classmethod
    def from_normalized_landmarks(cls, normalized_landmarks: list[NormalizedLandmark]) -> 'Keypoints':
        '''
        Construct object from detection results
        :param results: native mediapipe results object
        :return: object
        '''
        return cls(
            *[normalized_landmarks[idx] for idx in itertools.chain(range(11, 17), range(23, 29))],
            normalized_landmarks
        )
        
    def get_presences(self, threshold : float = 0.3) -> dict[str, bool]:
        """
        Returns a dictionary of the presence of each keypoint.
        True if the keypoint is present, False otherwise.
        
        :param threshold: The threshold for the presence of a keypoint
        :return: A dictionary of the presence of each keypoint
        """
        dict = self.to_dict()
        for key in dict:
            dict[key] = dict[key].presence > threshold
        return dict
    
    def is_in_frame(self, threshold: int = 0.3, match_ref = False, ref_keypoints: 'Keypoints' = None) -> tuple[int, int]:
        top = bottom = False
        presences = self.get_presences(threshold)

        if match_ref:
            present_ref_points = {name: kp for name, kp in ref_keypoints.to_dict().items() if ref_keypoints.get_presences()[name]}

            lowest_ref_point, _ = max((name, kp.y) for name, kp in present_ref_points.items())
            highest_ref_point, _ = min((name, kp.y) for name, kp in present_ref_points.items())

            top = presences[highest_ref_point]
            bottom = presences[lowest_ref_point]

        else:
            top = presences["left_shoulder"] and presences["right_shoulder"]
            bottom = presences["left_ankle"] and presences["right_ankle"]
        
        return int(top), int(bottom)
            




