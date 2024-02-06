from score import Score
from pose_estimation.keypoint_statistics import KeypointStatistics
import numpy as np


class EuclideanScore(Score):

    # overriding abstract method 
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics, weights: np.ndarray = None) -> float:
        """
        Returns the average euclidean distance between 
        the landmarks of the two keypoints
        :param first_keypoints: first set of keypoints
        :param second_keypoints: second set of keypoints
        :param weights: weights to apply to each distance
        :return: euclidean score
        """
        
        first_landmarks = first_keypoints.to_numpy_positions()
        second_landmarks = second_keypoints.to_numpy_positions()

        dist_difference = np.linalg.norm(first_landmarks - second_landmarks, axis=1)
        if weights is not None:
            dist_difference = dist_difference * weights

        return np.mean(dist_difference)
