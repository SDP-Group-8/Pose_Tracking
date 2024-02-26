

from math import exp
from matplotlib import pyplot as plt
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

def grade_gradients2(ref_stats: list[KeypointStatistics], live_stats: list[KeypointStatistics], current_frame: int, score_func: Score = AngleScore, seg_length: int = 5) -> float:
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

    return np.mean([gradient_diff[1], gradient_diff[3]])

def scale(score: float) -> float:
    score = np.abs(score)
    score = np.rad2deg(score)

    L = 200 # max score * 2
    k = -0.0732 # logistic growth rate

    return L // (1 + exp(-k*score))

def detect_movement(live_stats: list[KeypointStatistics], current_frame: int,  seg_length: int = 10) -> bool:
    if current_frame == 0:
        return True
    differences = []
    min_frame_idx = max(current_frame-seg_length, 1)
    for i in range(min_frame_idx, current_frame+1):
        differences.append(AngleScore.compute_each_score(live_stats[i-1], live_stats[i]))
    avg_diff = np.mean(np.mean(differences, axis=0))
    return avg_diff >= 0.17


if __name__=="__main__":
    media_pipe_video = MediaPipeVideo("dances/renegade.mp4")
    ref_pose_data = media_pipe_video.estimate_video()
    ref_stats = [KeypointStatistics.from_keypoints(pose) for pose in ref_pose_data]

    live_video = MediaPipeVideo("dances/ddd.mov")
    live_data = live_video.estimate_video()
    live_stats = [KeypointStatistics.from_keypoints(pose) for pose in live_data]
    live_stats = live_stats[:len(ref_stats)]
    

    from pose_estimation.scoring.calc_weights import compute_weights_comparison
    gradients = [grade_gradients(ref_stats, live_stats, i) for i in range(len(ref_stats))]
    weighted_scores = [AngleScore.compute_score(ref_stats[i], live_stats[i], isScaled=True, weights=compute_weights_comparison(ref_stats, live_stats, i, seg_length=10))/100 for i in range(len(ref_stats))]
    isMoving = [detect_movement(live_stats, i) for i in range(len(ref_stats))]

    x = range(len(ref_stats))
    plt.plot(x, gradients, label= "gradient difference")
    plt.plot(x, weighted_scores, label= "scores with weights")
    plt.plot(x, isMoving, label="is live video moving")
    plt.legend()
    plt.show()    



