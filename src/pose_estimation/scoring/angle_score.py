from __future__ import annotations
from pose_estimation.scoring.score import Score
from pose_estimation.keypoint_statistics import KeypointStatistics
import numpy as np
from math import exp


class AngleScore(Score):
    # overriding abstract method 
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                      weights: np.ndarray = None, isScaled : bool = False) -> float:
        """
        Returns the average angle difference between
        two keypoints statistics
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each angle
        :return: angle score
        """
        angle_diff = np.abs(first_keypoints.to_numpy() - second_keypoints.to_numpy())
        if isScaled:
            angle_diff = [AngleScore.scale_score(score) for score in angle_diff]
        if weights is not None:
            angle_diff = angle_diff * weights
        
        return np.mean(angle_diff)
    
    def compute_each_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                           weights: np.ndarray = None, isScaled : bool = False) -> list[float]:
        """
        Returns the angle difference between
        the two keypoints statistics for each angle
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each angle
        :return: array of each angle difference
        """
        angle_diff = np.abs(first_keypoints.to_numpy() - second_keypoints.to_numpy())

        if isScaled:
            angle_diff = [AngleScore.scale_score(score) for score in angle_diff]
        if weights is not None:
            angle_diff = angle_diff * weights
        return angle_diff


    def scale_score(angle_diff : float) -> float:
        """
        Returns the scaled score of the angle difference
        :param angle_diff: angle difference
        :return: scaled score
        """
        angle_diff = np.abs(angle_diff)
        angle_diff = np.rad2deg(angle_diff)

        L = 200 # max score * 2
        k = -0.0732 # logistic growth rate

        return L // (1 + exp(-k*angle_diff))


