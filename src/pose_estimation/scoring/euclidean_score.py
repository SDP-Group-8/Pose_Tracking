from score import Score
from keypoint_statistics import KeypointStatistics
import numpy as np


class EuclideanScore(Score):

    # overriding abstract method 
    def compute_score(first_keypoints: KeypointStatistics, second_keypoints: KeypointStatistics) -> float:
        """
        Returns the average euclidean distance between 
        the landmarks of the two keypoints
        """
        first_landmarks = np.array(first_keypoints.keypoints.normalized_landmarks)
        second_landmarks = np.array(second_keypoints.keypoints.normalized_landmarks)

        return np.mean(np.abs(first_landmarks - second_landmarks))

    