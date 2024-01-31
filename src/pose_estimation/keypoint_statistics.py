import math

from keypoints import Keypoints
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from dataclasses import dataclass

@dataclass
class KeypointStatistics:
    keypoints: Keypoints

    r_shoulder_l_shoulder_l_elbow: float
    l_shoulder_l_elbow_l_wrist: float

    r_shoulder_l_shoulder_l_elbow: float
    r_shoulder_r_elbow_r_wrist: float

    r_hip_l_hip_l_knee: float
    l_hip_l_knee_l_ankle: float

    l_hip_r_hip_r_knee: float
    r_hip_r_knee_r_ankle: float

    def to_list(self) -> [float]:
        """
        Converts class into a list of angles.
        :return: List of key angles.
        """
        return [
            self.r_shoulder_l_shoulder_l_elbow,
            self.l_shoulder_l_elbow_l_wrist,
            self.r_shoulder_l_shoulder_l_elbow,
            self.r_shoulder_r_elbow_r_wrist,
            self.r_hip_l_hip_l_knee,
            self.l_hip_l_knee_l_ankle,
            self.l_hip_r_hip_r_knee,
            self.r_hip_r_knee_r_ankle
        ]

    @classmethod
    def from_keypoints(cls, keypoints: Keypoints) -> 'KeypointStatistics':
        """
        Calculates the angle abc for key dance points on body for a given frame.
        :param keypoints: Holds key body points for a given frame.
        :return: Itself, which holds the angle data.
        """

        r_shoulder_l_shoulder_l_elbow = KeypointStatistics.calculate_angle(
            keypoints.right_shoulder,
            keypoints.left_shoulder,
            keypoints.l_shoulder_l_elbow_l_wrist
        )

        l_shoulder_l_elbow_l_wrist = KeypointStatistics.calculate_angle(
            keypoints.left_shoulder,
            keypoints.l_shoulder_l_elbow_l_wrist,
            keypoints.left_wrist
        )

        r_shoulder_l_shoulder_l_elbow = KeypointStatistics.calculate_angle(
            keypoints.left_shoulder,
            keypoints.right_shoulder,
            keypoints.r_shoulder_r_elbow_r_wrist
        )

        r_shoulder_r_elbow_r_wrist = KeypointStatistics.calculate_angle(
            keypoints.right_shoulder,
            keypoints.r_shoulder_r_elbow_r_wrist,
            keypoints.right_wrist
        )

        r_hip_l_hip_l_knee = KeypointStatistics.calculate_angle(
            keypoints.l_hip_r_hip_r_knee,
            keypoints.r_hip_l_hip_l_knee,
            keypoints.l_hip_l_knee_l_ankle
        )

        l_hip_l_knee_l_ankle = KeypointStatistics.calculate_angle(
            keypoints.r_hip_l_hip_l_knee,
            keypoints.l_hip_l_knee_l_ankle,
            keypoints.left_ankle
        )

        l_hip_r_hip_r_knee = KeypointStatistics.calculate_angle(
            keypoints.r_hip_l_hip_l_knee,
            keypoints.l_hip_r_hip_r_knee,
            keypoints.r_hip_r_knee_r_ankle
        )

        r_hip_r_knee_r_ankle = KeypointStatistics.calculate_angle(
            keypoints.l_hip_r_hip_r_knee,
            keypoints.r_hip_r_knee_r_ankle,
            keypoints.right_ankle
        )

        return cls(
            keypoints,
            r_shoulder_l_shoulder_l_elbow,
            l_shoulder_l_elbow_l_wrist,
            r_shoulder_l_shoulder_l_elbow,
            r_shoulder_r_elbow_r_wrist,
            r_hip_l_hip_l_knee,
            l_hip_l_knee_l_ankle,
            l_hip_r_hip_r_knee,
            r_hip_r_knee_r_ankle
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
