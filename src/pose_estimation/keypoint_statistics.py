from pose_estimation.keypoints import Keypoints

class KeypointStatistics:
    keypoints: Keypoints

    r_shoulder_l_shoulder_l_elbow: float
    l_shoulder_l_elbow_l_wrist: float

    r_shoulder_l_shoulder_l_elbow: float
    l_shoulder_l_elbow_l_wrist: float
    
    l_shoulder_r_shoulder_r_elbow: float
    r_shoulder_r_elbow_r_wrist: float

    r_hip_l_hip_l_knee: float
    l_hip_l_knee_l_ankle: float

    r_hip_l_knee_l_knee: float
    l_hip_l_knee_l_ankle: float
    
    l_hip_r_hip_r_knee: float
    r_hip_r_knee_r_ankle: float

    def to_list():
        pass

    def from_keypoints():
        pass