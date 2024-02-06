from copy import deepcopy
from pose_estimation.keypoint_statistics import KeypointStatistics
from scipy.spatial.transform import Rotation
import numpy as np

class KeypointScaling:
    @staticmethod
    def calculate_scaling_factor(ref_landmark, live_landmark):
        reference_distance = KeypointScaling.calculate_distance(ref_landmark.keypoints.left_shoulder, ref_landmark.keypoints.right_shoulder)
        live_distance = KeypointScaling.calculate_distance(live_landmark.keypoints.left_shoulder, live_landmark.keypoints.right_shoulder)
        return reference_distance / live_distance

    @staticmethod
    def calculate_distance(a, b):
        return np.linalg.norm(np.asarray([a.x - b.x, a.y - b.y]))

    @staticmethod
    def scale_keypoint_distance(key_point1, key_point2, scaling_factor):
        return KeypointScaling.calculate_distance(key_point1, key_point2) * scaling_factor

    @staticmethod
    def compute_scaled_keypoints(key_point1, key_point2, angle, scaling_factor):
        delta = np.asarray([key_point1.x - key_point2.x, key_point1.y - key_point2.y])
        rotated = Rotation.from_rotvec([0, 0, angle]).as_matrix()[:2, :2] @ delta
        distance = np.linalg.norm(delta)

        return (scaling_factor * rotated / distance) + np.asarray([key_point2.x, key_point2.y])

    @staticmethod
    def scale_keypoints(ref_landmark: KeypointStatistics, live_landmark: KeypointStatistics):
        scaling_factor = KeypointScaling.calculate_scaling_factor(ref_landmark, live_landmark)

        for (angle_name, live_endpoints), reference_endpoints in \
            zip(live_landmark.property_map.items(), ref_landmark.property_map.values()):

            scaled_coordinates = KeypointScaling.compute_scaled_keypoints(
                *live_endpoints[:2],
                vars(live_landmark)[angle_name],
                KeypointScaling.scale_keypoint_distance(
                    *reference_endpoints[1:],
                    scaling_factor
                )
            )

            live_endpoints[2].x, live_endpoints[2].y = tuple(scaled_coordinates)
        
        return live_landmark
