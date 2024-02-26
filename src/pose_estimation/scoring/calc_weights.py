


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

def compute_weights_comparison(ref_stats: list[KeypointStatistics], live_stats: list[KeypointStatistics], current_frame: int, score_func : Score = AngleScore, seg_length : int = 5) -> list[float]:
    if current_frame == 0:
        return 0

    # Initialize lists to store the differences
    ref_differences = []
    live_differences = []
    
    # Determine the starting frame index, ensuring it's within bounds
    min_frame_idx = max(current_frame-seg_length, 1)
    
    for i in range(min_frame_idx, current_frame+1):
        ref_differences.append(score_func.compute_each_score(ref_stats[i-1], ref_stats[i]))
        live_differences.append(score_func.compute_each_score(live_stats[i-1], live_stats[i]))
    ref_differences = np.array(ref_differences)
    live_differences = np.array(live_differences)
    
    ref_gradients = np.mean(ref_differences, axis=0)
    live_gradients = np.mean(live_differences, axis=0)
    gradient_diff = np.abs(ref_gradients - live_gradients)
    gradient_diff[gradient_diff <= 0.17] = 0
    print(norm_weights(gradient_diff))
    return norm_weights(gradient_diff)

def norm_weights(weights: list[float]) -> list[float]:
    scale = len(weights) / sum(weights)
    return [weight * scale for weight in weights]


