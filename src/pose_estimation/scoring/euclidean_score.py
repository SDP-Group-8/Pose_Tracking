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

        def get_x_y(landmarks):
            x_y_results = np.empty([len(landmarks), 2])
            for i in range(len(landmarks)):
                x_y_results[i] = [landmarks[i].x, landmarks[i].y]
            return x_y_results
        
        first_landmarks = np.array(first_keypoints.keypoints.normalized_landmarks)
        first_coords = get_x_y(first_landmarks)

        second_landmarks = np.array(second_keypoints.keypoints.normalized_landmarks)
        second_coords = get_x_y(second_landmarks)

        dist_difference = np.linalg.norm(first_coords - second_coords, axis=1)
        if weights is not None:
            dist_difference = dist_difference * weights

        return np.mean(dist_difference)




    