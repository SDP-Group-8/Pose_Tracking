import math

from keypoints import Keypoints
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from dataclasses import dataclass


@dataclass
class KeypointStatistics:
    keypoints: Keypoints

    left_armpit: float
    left_elbow: float

    right_armpit: float
    right_elbow: float

    left_hip: float
    left_knee: float

    right_hip: float
    right_knee: float

    def to_list(self) -> [float]:
        """
        Converts class into a list of angles.
        :return: List of key angles.
        """
        return [
            self.left_armpit,
            self.left_elbow,
            self.right_armpit,
            self.right_elbow,
            self.left_hip,
            self.left_knee,
            self.right_hip,
            self.right_knee
        ]

    @classmethod
    def from_keypoints(cls, keypoints: Keypoints) -> 'KeypointStatistics':
        """
        Calculates the angle abc for key dance points on body for a given frame.
        :param keypoints: Holds key body points for a given frame.
        :return: Itself, which holds the angle data.
        """

        left_armpit = KeypointStatistics.calculate_angle(
            keypoints.right_shoulder,
            keypoints.left_shoulder,
            keypoints.left_elbow
        )

        left_elbow = KeypointStatistics.calculate_angle(
            keypoints.left_shoulder,
            keypoints.left_elbow,
            keypoints.left_wrist
        )

        right_armpit = KeypointStatistics.calculate_angle(
            keypoints.left_shoulder,
            keypoints.right_shoulder,
            keypoints.right_elbow
        )

        right_elbow = KeypointStatistics.calculate_angle(
            keypoints.right_shoulder,
            keypoints.right_elbow,
            keypoints.right_wrist
        )

        left_hip = KeypointStatistics.calculate_angle(
            keypoints.right_hip,
            keypoints.left_hip,
            keypoints.left_knee
        )

        left_knee = KeypointStatistics.calculate_angle(
            keypoints.left_hip,
            keypoints.left_knee,
            keypoints.left_ankle
        )

        right_hip = KeypointStatistics.calculate_angle(
            keypoints.left_hip,
            keypoints.right_hip,
            keypoints.right_knee
        )

        right_knee = KeypointStatistics.calculate_angle(
            keypoints.right_hip,
            keypoints.right_knee,
            keypoints.right_ankle
        )

        return cls(
            keypoints,
            left_armpit,
            left_elbow,
            right_armpit,
            right_elbow,
            left_hip,
            left_knee,
            right_hip,
            right_knee
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
        ab = (xb - xa, yb - ya)
        bc = (xc - xb, yc - yb)

        # Calculate dot product of AB and BC
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]

        # Calculate magnitudes of AB and BC
        magnitude_AB = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        magnitude_BC = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        # Calculate cosine of the angle
        cos = dot_product / (magnitude_AB * magnitude_BC)

        return math.degrees(math.acos(cos))
