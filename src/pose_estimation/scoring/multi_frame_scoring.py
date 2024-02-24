

import numpy as np
from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.scoring.angle_score import AngleScore
from pose_estimation.scoring.score import Score


def grade_gradients(ref_stats: list[KeypointStatistics], live_stats: list[KeypointStatistics], score_func: Score = AngleScore, current_frame: int = 0, seg_length: int = 5) -> list[float]:
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
    # Initialize lists to store the differences
    ref_differences = []
    live_differences = []

    # Determine the starting frame index, ensuring it's within bounds
    min_frame_idx = max(current_frame-seg_length, 1)

    for i in range(min_frame_idx, current_frame+1):
        ref_differences.append(score_func.compute_each_score(ref_stats[i-1], ref_stats[i]))
        live_differences.append(score_func.compute_each_score(live_stats[i-1], live_stats[i]))

    if current_frame < seg_length:
        ref_differences.insert(0, np.zeros(ref_differences[0].shape))
        live_differences.insert(0, np.zeros(live_differences[0].shape))
    
    ref_gradients = np.mean(ref_differences, axis=0)
    live_gradients = np.mean(live_differences, axis=0) 
    return [score_func.scale_score(score) for score in (live_gradients - ref_gradients)]
    

