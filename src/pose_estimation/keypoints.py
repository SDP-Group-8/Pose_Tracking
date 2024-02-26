from dataclasses import dataclass
import itertools
import numpy as np

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

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
    def get_presences(self) -> np.ndarray[bool]:
        """
        Returns a boolean array of the presence of each keypoint.
        """
        npkeypoints = np.array([
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
            ])
        presences = np.vectorize(lambda x: x.presence)(npkeypoints)
        return presences > 0.8