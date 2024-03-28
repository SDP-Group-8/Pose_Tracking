from __future__ import annotations

from pose_estimation.scoring.score import Score
from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.keypoints import Keypoints
import numpy as np


class EuclideanScore(Score):

    # overriding abstract method 
    def compute_score(self, first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, 
                      weights: np.ndarray = None) -> float:
        """
        Returns the average euclidean distance between 
        the landmarks of the two keypoints
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each distance
        :return: euclidean score
        """
        return np.mean(list(self.compute_each_score(first_keypoints, second_keypoints, weights).values()))

    def compute_each_score(self, first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics,
                            weights: np.ndarray = None) -> dict[str, float]:
          """
          Returns the euclidean distance between
          the landmarks of the two keypoints for each landmark
          :param first_keypoints: first set of keypoints
          :param second_keypoints: second set of keypoints
          :param weights: weights to apply to each distance
          :return: array of each euclidean distance
          """
          first_landmarks = first_keypoints.keypoints.to_numpy_positions()[:, :2]
          second_landmarks = second_keypoints.keypoints.to_numpy_positions()[:, :2]
    
          dist_difference = np.linalg.norm(first_landmarks - second_landmarks, axis=1)
          if weights is not None:
                dist_difference = dist_difference * weights

          return {f"{name}": score for name, score in zip(Keypoints.ordered_fields, dist_difference)}
