

import numpy as np
from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.scoring.angle_score import AngleScore


class DelayAngleScore:
    count = 0
    score_total = 0
    prev_score = 0

    def __init__(self, init_score):
        self.count = 0
        self.score_total = 0
        self.prev_score = init_score


    def compute_score(self, ref_stats: KeypointStatistics, live_stats: KeypointStatistics, 
                      weights: np.ndarray = None, isScaled : bool = False, seg_length=20) -> float:
        
        

        score = AngleScore.compute_score(ref_stats, live_stats, weights, isScaled)
        self.score_total += score
        self.count += 1

        if self.count == seg_length:
            if self.count !=0:
                finalScore = self.score_total / self.count
            else:
                finalScore = score
            self.count = 0
            self.score_total = 0
            self.prev_score = finalScore
            return finalScore

        return self.prev_score