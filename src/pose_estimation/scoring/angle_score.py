from math import exp
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
        angle_diff = np.abs(first_keypoints.to_numpy() - second_keypoints.to_numpy())
        if weights is not None:
            angle_diff = angle_diff * weights

        return np.mean(angle_diff)
    
    def compute_scaled_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, weights: np.ndarray = None) -> float:
        """
        Returns the scaled score based on angle difference between
        the two keypoints
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each angle
        :return: scaled angle score
        """
        angle_diff = np.abs(second_keypoints.to_numpy() - first_keypoints.to_numpy())
        angle_diff = np.rad2deg(angle_diff)

        L = 200 # max score * 2
        k = -0.0732 # logistic growth rate

        log_func = lambda x: L / (1 + exp(-k*x)) 
        log_score = [log_func(angle) for angle in angle_diff] # apply logistic function to each angle

        if weights is not None:
            log_score = log_score * weights
        return np.mean(log_score)
