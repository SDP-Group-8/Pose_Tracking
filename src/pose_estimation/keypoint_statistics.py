import math
import numpy as np

from pose_estimation.keypoints import Keypoints
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from dataclasses import dataclass

from pose_estimation.pre_processing.orientation import Orientation

@dataclass
class KeypointStatistics:
    keypoints: Keypoints
    property_map: dict[str]
    orientation: Orientation

    r_shoulder_l_shoulder_l_elbow: float
    l_shoulder_l_elbow_l_wrist: float

    l_shoulder_r_shoulder_r_elbow: float
    r_shoulder_r_elbow_r_wrist: float

    r_hip_l_hip_l_knee: float
    l_hip_l_knee_l_ankle: float

    l_hip_r_hip_r_knee: float
    r_hip_r_knee_r_ankle: float

    @staticmethod
    def create_map(keypoints: Keypoints):
        return {
            'r_shoulder_l_shoulder_l_elbow': [keypoints.right_shoulder, keypoints.left_shoulder, keypoints.left_elbow],
            'l_shoulder_l_elbow_l_wrist': [keypoints.left_shoulder, keypoints.left_elbow, keypoints.left_wrist],
            'l_shoulder_r_shoulder_r_elbow': [keypoints.left_shoulder, keypoints.right_shoulder, keypoints.right_elbow],
            'r_shoulder_r_elbow_r_wrist': [keypoints.right_shoulder, keypoints.right_elbow, keypoints.right_wrist],
            'r_hip_l_hip_l_knee': [keypoints.right_hip, keypoints.left_hip, keypoints.left_knee],
            'l_hip_l_knee_l_ankle': [keypoints.left_hip, keypoints.left_knee, keypoints.left_ankle],
            'l_hip_r_hip_r_knee': [keypoints.left_hip, keypoints.right_hip, keypoints.right_knee],
            'r_hip_r_knee_r_ankle': [keypoints.right_hip, keypoints.right_knee,keypoints.right_ankle]
        }

    def to_numpy(self) -> 'np.ndarray':
        """
        Converts class into a list of angles.
        :return: List of key angles.
        """
        return np.array([
            self.r_shoulder_l_shoulder_l_elbow,
            self.l_shoulder_l_elbow_l_wrist,
            self.l_shoulder_r_shoulder_r_elbow,
            self.r_shoulder_r_elbow_r_wrist,
            self.r_hip_l_hip_l_knee,
            self.l_hip_l_knee_l_ankle,
            self.l_hip_r_hip_r_knee,
            self.r_hip_r_knee_r_ankle
        ])

    @classmethod
    def from_keypoints(cls, keypoints: Keypoints) -> 'KeypointStatistics':
        """
        Calculates the angle abc for key dance points on body for a given frame.
        :param keypoints: Holds key body points for a given frame.
        :return: Itself, which holds the angle data.
        """
        body_map = KeypointStatistics.create_map(keypoints)

        return cls(
            keypoints,
            body_map,
            Orientation.from_keypoints(keypoints),
            *(KeypointStatistics.calculate_angle(*parts) for parts in body_map.values())
        )

    @staticmethod
    def calculate_angle(point_a: NormalizedLandmark, point_b: NormalizedLandmark,
                        point_c: NormalizedLandmark) -> float:
        """
        Calculates an angle abc.
        :param point_a: Point a.
        :param point_b: Point b.
        :param point_c: Point c.
        :return: Angle ABC.
        """
        xa = point_a.x
        ya = point_a.y

        xb = point_b.x
        yb = point_b.y

        xc = point_c.x
        yc = point_c.y

        # Calculate vectors AB and BC
        ab = np.array([xb - xa, yb - ya])
        bc = np.array([xb - xc, yb - yc])

        # Calculate dot product of AB and BC
        dot_product = np.dot(ab, bc)
        determinant = np.linalg.det(np.vstack([ab, bc]))

        # Calculate and return angle.
        return np.arctan2(determinant, dot_product)
    