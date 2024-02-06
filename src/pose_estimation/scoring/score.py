from abc import ABC, abstractmethod
import numpy as np
from pose_estimation.keypoint_statistics import KeypointStatistics


class Score(ABC):
    @abstractmethod
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, weights: np.ndarray = None) -> float:
        pass