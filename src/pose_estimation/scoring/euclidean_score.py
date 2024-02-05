from pose_estimation.scoring.score import Score
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
        first_landmarks = np.array(first_keypoints.keypoints.normalized_landmarks)
        second_landmarks = np.array(second_keypoints.keypoints.normalized_landmarks)

        dist_difference = np.linalg.norm(first_landmarks - second_landmarks)
        if weights is not None:
            dist_difference = dist_difference * weights

        return np.mean(dist_difference)


    