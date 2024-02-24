


from matplotlib import pyplot as plt
import numpy as np

from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.mediapipe import MediaPipe
from pose_estimation.mediapipe_video import MediaPipeVideo
from pose_estimation.scoring.euclidean_score import EuclideanScore
from pose_estimation.scoring.score import Score
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from pose_estimation.scoring.angle_score import AngleScore


def compute_weights(ref_dataset: list[KeypointStatistics], score_func : Score = AngleScore, seg_length : int = 5) -> np.ndarray[np.ndarray[float]]:
    """
    takes in a list of KeypointStatistics objects, and
    returns a list of weights for each frame, with
    higher weights for parts that change more

    :param ref_dataset: array of reference keypoint statistics
    :param score_func: scoring function to use
    :param seg_length: length of the before and after segment to use for computing the gradients
    :return: array of weights for each frame
    """

    weights = []

    differences = []
    # Loop through the reference dataset to compute score differences between consecutive frames
    for i in range(1, len(ref_dataset)):
        differences.append(
            score_func.compute_each_score(ref_dataset[i-1], ref_dataset[i]))
    differences.insert(0, np.zeros(differences[0].shape))
    differences = np.array(differences)
    
    # Initialize an empty list to store the gradients
    scoreGradients = []
    
    for i in range(0, len(differences)):
        # Calculate the start and end index of the segment
        startIdx = max(0, i - seg_length)  
        endIdx = min(len(differences), i + seg_length)
        
        # Calculate the mean difference over the segment and append to 'scoreGradients'
        total = np.mean(differences[startIdx:endIdx], axis=0)
        scoreGradients.append(total)
    scoreGradients = np.array(scoreGradients)

    # Normalize the gradients based on each column
    weights = scoreGradients / np.linalg.norm(scoreGradients, 1, axis=0, keepdims=True)

    # Further normalize each set of weights individually
    weights = np.array(list(map(norm_weights, weights)))
    return weights

def norm_weights(weights: list[float]) -> list[float]:
    scale = len(weights) / sum(weights)
    return [weight * scale for weight in weights]


