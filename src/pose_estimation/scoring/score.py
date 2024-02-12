from abc import ABC, abstractmethod
import numpy as np
from pose_estimation.keypoint_statistics import KeypointStatistics


class Score(ABC):
    #implement abstract method based on the AngleScore and EuclideanScore classes, the compute_score, scale_score and compute_each_score funcs
    @abstractmethod
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                      weights: np.ndarray = None, isScaled : bool = False) -> float:
        pass

    @abstractmethod
    def compute_each_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                           weights: np.ndarray = None, isScaled : bool = False) -> list[float]:
        pass

    @abstractmethod
    def scale_score(score : float) -> float:
        pass
    