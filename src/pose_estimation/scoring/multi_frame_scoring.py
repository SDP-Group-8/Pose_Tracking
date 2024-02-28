

from math import exp
import numpy as np
from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.mediapipe_video import MediaPipeVideo
from pose_estimation.scoring.angle_score import AngleScore
from pose_estimation.scoring.score import Score


def grade_gradients(ref_stats: list[KeypointStatistics], live_stats: list[KeypointStatistics], current_frame: int, score_func: Score = AngleScore, seg_length: int = 5) -> float:
    """
    takes two lists of KeypointStatistics objects, and
    compares the rate of the keypoints changing over the last few
    frames in each list

    :param ref_stats: list of reference keypoint statistics
    :param live_stats: list of live keypoint statistics
    :param score_func: scoring class to use
    :param current_frame: index of the current frame
    :param seg_length: number of frames to use to compare the gradients
    :return: scaled score difference of the gradient
    """
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

    return np.mean(gradient_diff)

def scale(score: float) -> float:
    score = np.abs(score)
    score = np.rad2deg(score)

    L = 200 # max score * 2
    k = -0.0732 # logistic growth rate

    return L // (1 + exp(-k*score))

def detect_movement(ref_stats: list[KeypointStatistics], live_stats: list[KeypointStatistics], current_frame: int,  seg_length: int = 5, threshold: int = 0.17) -> bool:
    if current_frame == 0:
        return True
    live_differences = []
    ref_differences = []
    live_presences = []
    ref_presences = []

    min_frame_idx = max(current_frame-seg_length, 1)
    for i in range(min_frame_idx, current_frame+1):
        live_differences.append(AngleScore.compute_each_score(live_stats[i-1], live_stats[i]))
        live_presences.append(live_stats[i].angle_presences() or ref_stats[i-1].angle_presences())
        
    for i in range(max(current_frame-30, 1), current_frame+1):
        ref_differences.append(AngleScore.compute_each_score(ref_stats[i-1], ref_stats[i]))
        ref_presences.append(ref_stats[i].angle_presences() or live_stats[i-1].angle_presences())

    live_avg_diff = np.ma.array(data=live_differences, mask=live_presences).mean(axis=0).mean(axis=0)
    ref_avg_diff = np.ma.array(data=ref_differences, mask=ref_presences).mean(axis=0).mean(axis=0)
    # live_avg_diff = np.mean(np.mean(live_differences, axis=0))
    # ref_avg_diff = np.mean(np.mean(ref_differences, axis=0))
    return live_avg_diff >= threshold or ref_avg_diff < threshold

    