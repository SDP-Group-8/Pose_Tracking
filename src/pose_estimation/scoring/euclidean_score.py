from __future__ import annotations

from math import exp
from pose_estimation.scoring.score import Score
from pose_estimation.keypoint_statistics import KeypointStatistics
import numpy as np


class EuclideanScore(Score):

    # overriding abstract method 
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                      weights: np.ndarray = None, isScaled : bool = False) -> float:
        """
        Returns the average euclidean distance between 
        the landmarks of the two keypoints
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each distance
        :return: euclidean score
        """
        
        first_landmarks = first_keypoints.keypoints.to_numpy_positions()
        second_landmarks = second_keypoints.keypoints.to_numpy_positions()

        dist_difference = np.linalg.norm(first_landmarks - second_landmarks, axis=1)
        if isScaled:
            dist_difference = [EuclideanScore.scale_score(score) for score in dist_difference]
        if weights is not None:
            dist_difference = dist_difference * weights
        
        return np.mean(dist_difference)

    def compute_each_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics,
                            weights: np.ndarray = None, isScaled : bool = False) -> list[float]:
          """
          Returns the euclidean distance between
          the landmarks of the two keypoints for each landmark
          :param first_keypoints: first set of keypoints
          :param second_keypoints: second set of keypoints
          :param weights: weights to apply to each distance
          :return: array of each euclidean distance
          """
          first_landmarks = first_keypoints.keypoints.to_numpy_positions()
          second_landmarks = second_keypoints.keypoints.to_numpy_positions()
    
          dist_difference = np.linalg.norm(first_landmarks - second_landmarks, axis=1)
          if isScaled:
                dist_difference = [EuclideanScore.scale_score(score) for score in dist_difference]
          if weights is not None:
                dist_difference = dist_difference * weights
          return dist_difference
    

    def scale_score(dist : float) -> float:
        """
        Returns the scaled score of the euclidean distance
        :param dist: distance between two landmarks
        :return: scaled score
        """
        dist = np.abs(dist)

        L = 200 # max score * 2
        k = -5.493 # logistic growth rate

        return L // (1 + exp(-k*dist))
