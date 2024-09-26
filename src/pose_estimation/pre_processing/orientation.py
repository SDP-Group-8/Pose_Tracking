
from dataclasses import dataclass
import math
import numpy as np
from pose_estimation.keypoints import Keypoints
from pose_estimation.keypoint_statistics import KeypointStatistics
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from scipy.spatial.transform import Rotation 

@dataclass
class Orientation:
    shoulder_orient: float
    hip_orient: float
    torso_orient: float
    whole_body_orient: float

    @classmethod
    def from_keypoints(cls, keypoints : Keypoints) -> 'Orientation':
        """
        gets rotation matrix
        """
        return cls(
            Orientation.get_angle(keypoints.right_shoulder, keypoints.left_shoulder),
            Orientation.get_angle(keypoints.right_hip, keypoints.left_hip),
            Orientation.get_angle(keypoints.right_shoulder, keypoints.left_hip),
            Orientation.get_angle(keypoints.right_shoulder, keypoints.left_ankle)
                   ) 

    @staticmethod
    def get_angle(origin: NormalizedLandmark, target: NormalizedLandmark) -> float:
        """
        get angle between two points in the x and z plane
        will be zero if the two points are parrelel to the camera

        :param origin: origin landmark
        :param target: target landmark
        :return: angle in degrees
        """

        origin_vect = np.array([origin.x, origin.y, origin.z])
        target_vect = np.array([target.x, target.y, target.z])
        vect = target_vect - origin_vect
        angle = np.arctan2(vect[2], vect[0])
        return np.rad2deg(angle)
        
    
    


    
        
