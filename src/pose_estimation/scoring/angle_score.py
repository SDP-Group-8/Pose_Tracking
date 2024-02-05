from pose_estimation.scoring.score import Score
from pose_estimation.keypoint_statistics import KeypointStatistics
import numpy as np


class AngleScore(Score):
    # overriding abstract method 
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, weights: np.ndarray = None) -> float:
        """
        Returns the average angle difference between
        the two keypoints
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each angle
        :return: angle score
        """
        angle_diff = np.abs(first_keypoints.to_list() - second_keypoints.to_list())
        if weights is not None:
            angle_diff = angle_diff * weights

        return np.mean(angle_diff)