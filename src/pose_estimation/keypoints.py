from dataclasses import dataclass
import itertools

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.landmark_detection_result import LandmarksDetectionResult

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

    def to_normalized_landmarks(self) -> [NormalizedLandmark]:
        '''
        Turn object into native mediapipe results object
        :return: mediapipe results object
        '''
        return self.results

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
