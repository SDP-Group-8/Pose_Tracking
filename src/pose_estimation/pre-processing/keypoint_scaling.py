import math
from pose_estimation.keypoint_statistics import KeypointStatistics
from pose_estimation.keypoints import Keypoints

class KeypointScaling:
    @staticmethod
    def calculate_scaling_factor(ref_landmark, live_landmark):
        return KeypointScaling.calculate_distance(ref_landmark.left_shoulder, ref_landmark.right_shoulder) /        KeypointScaling.calculate_distance(live_landmark.left_shoulder, live_landmark.right_shoulder)

    @staticmethod
    def calculate_distance(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def scale_keypoint_distance(key_point1, key_point2, scaling_factor):
        return KeypointScaling.calculate_distance(key_point1, key_point2) * scaling_factor

    @staticmethod
    def compute_point(x1, y1, x2, y2, theta, k):
        dx = x1 - x2
        dy = y1 - y2
        rdx = dx * math.cos(theta) - dy * math.sin(theta)
        rdy = dx * math.sin(theta) + dy * math.cos(theta)
        ab = math.sqrt(dx*dx + dy*dy)
        return (k/ab * rdx + x2, k/ab * rdy + y2)

    @staticmethod
    def compute_scaled_new_keypoint(key_point1, key_point2, angle, scaling_factor):
        return KeypointScaling.compute_point(key_point1.x, key_point1.y, key_point2.x, key_point2.y, angle, scaling_factor)

    @staticmethod
    def update_keypoint(keypoint, new_keypoint):
        keypoint.x, keypoint.y = new_keypoint

    """
    Main function to scale the keypoints
    """
    @staticmethod
    def scale_keypoint(ref_landmark: Keypoints, live_landmark: Keypoints):
        scaling_factor = KeypointScaling.calculate_scaling_factor(ref_landmark, live_landmark)

        # Left Elbow
        KeypointScaling.update_keypoint(
            live_landmark.left_elbow,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.right_shoulder,
                live_landmark.left_shoulder,
                KeypointStatistics.r_shoulder_l_shoulder_l_elbow,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.left_shoulder,
                    ref_landmark.left_elbow,
                    scaling_factor
                )
            )
        )

        # Left Wrist
        KeypointScaling.update_keypoint(
            live_landmark.left_wrist,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.left_shoulder,
                live_landmark.left_elbow,
                KeypointStatistics.l_shoulder_l_elbow_l_wrist,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.left_elbow,
                    ref_landmark.left_wrist,
                    scaling_factor
                )
            )
        )

        # Right Elbow
        KeypointScaling.update_keypoint(
            live_landmark.right_elbow,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.left_shoulder,
                live_landmark.right_shoulder,
                KeypointStatistics.l_shoulder_r_shoulder_r_elbow,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.right_shoulder,
                    ref_landmark.right_elbow,
                    scaling_factor
                )
            )
        )

        # Right Wrist
        KeypointScaling.update_keypoint(
            live_landmark.right_wrist,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.right_shoulder,
                live_landmark.right_elbow,
                KeypointStatistics.r_shoulder_r_elbow_r_wrist,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.right_elbow,
                    ref_landmark.right_wrist,
                    scaling_factor
                )
            )
        )

        # Left Knee
        KeypointScaling.update_keypoint(
            live_landmark.left_knee,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.right_hip,
                live_landmark.left_hip,
                KeypointStatistics.r_hip_l_hip_l_knee,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.left_hip,
                    ref_landmark.left_knee,
                    scaling_factor
                )
            )
        )
        
        # Left Ankle
        KeypointScaling.update_keypoint(
            live_landmark.left_ankle,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.left_hip,
                live_landmark.left_knee,
                KeypointStatistics.l_hip_l_knee_l_ankle,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.left_knee,
                    ref_landmark.left_ankle,
                    scaling_factor
                )
            )
        )
        
        # Right Knee
        KeypointScaling.update_keypoint(
            live_landmark.right_knee,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.left_hip,
                live_landmark.right_hip,
                KeypointStatistics.l_hip_r_hip_r_knee,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.right_hip,
                    ref_landmark.right_knee,
                    scaling_factor
                )
            )
        )
        
        # Right Ankle
        KeypointScaling.update_keypoint(
            live_landmark.right_ankle,
            KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.right_hip,
                live_landmark.right_knee,
                KeypointStatistics.r_hip_r_knee_r_ankle,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.right_knee,
                    ref_landmark.right_ankle,
                    scaling_factor
                )
            )
        )
        
        
