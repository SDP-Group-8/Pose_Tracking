from __future__ import annotations

from pose_estimation.scoring.score import Score
from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.keypoints import Keypoints

import numpy as np

class AngleScore(Score):
    # overriding abstract method 
    def compute_score(self, first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                      weights: np.ndarray = None) -> float:
        """
        Returns the average angle difference between
        two keypoints statistics
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each angle
        :return: angle score
        """
        return np.mean(list(self.compute_each_score(first_keypoints, second_keypoints, weights).values()))
    
    def compute_each_score(self, first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                           weights: np.ndarray = None) -> dict[str, float]:
        """
        Returns the angle difference between
        the two keypoints statistics for each angle
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each angle
        :return: array of each angle difference
        """
        angle_diff = np.abs(first_keypoints.to_numpy() - second_keypoints.to_numpy())

        if weights is not None:
            angle_diff = angle_diff * weights
    
        return {f"{name}": score for name, score in zip(Keypoints.ordered_fields, angle_diff)}

