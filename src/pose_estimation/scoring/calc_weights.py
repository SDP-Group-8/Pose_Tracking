from __future__ import annotations


import numpy as np
from typing import List

from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.scoring import score
from pose_estimation.scoring.angle_score import AngleScore


def compute_weights(ref_dataset: List[KeypointStatistics], score_func : Score = AngleScore, seg_length : int = 5):
    """
    TODO: Implement function to compute weights for each landmark / angle using ref video
    TODO: Take into account visibility + degree of change 
    :param ref_dataset: array of reference keypoint statistics
    :param score_func: scoring function to use
    :return: array of weights
    """
    
    return []