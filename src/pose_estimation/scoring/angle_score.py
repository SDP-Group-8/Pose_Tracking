from score import Score
from keypoint_statistics import KeypointStatistics
import numpy as np


class AngleScore(Score):
    # overriding abstract method 
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics) -> float:
        """
        Returns the average angle difference between
        the landmarks of the two keypoints
        """
        angle_diff = np.abs(first_keypoints.to_list() - second_keypoints.to_list())
        return np.mean(angle_diff)