import math
from copy import deepcopy
from pose_estimation.keypoints import Keypoints

class KeypointScaling:
    @staticmethod
    def calculate_scaling_factor(ref_landmark, live_landmark):
        return KeypointScaling.calculate_distance(ref_landmark.keypoints.left_shoulder, ref_landmark.keypoints.right_shoulder) /        KeypointScaling.calculate_distance(live_landmark.keypoints.left_shoulder, live_landmark.keypoints.right_shoulder)

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

        scaledKeypoint = deepcopy(live_landmark)

        # Left Elbow
        scaledKeypoint.keypoints.left_shoulder.x, scaledKeypoint.keypoints.left_shoulder.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.right_shoulder,
                live_landmark.keypoints.left_shoulder,
                live_landmark.r_shoulder_l_shoulder_l_elbow,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.left_shoulder,
                    ref_landmark.keypoints.left_elbow,
                    scaling_factor
                )
            )

        # Left Wrist
        scaledKeypoint.keypoints.left_wrist.x, scaledKeypoint.keypoints.left_wrist.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.left_shoulder,
                live_landmark.keypoints.left_elbow,
                live_landmark.l_shoulder_l_elbow_l_wrist,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.left_elbow,
                    ref_landmark.keypoints.left_wrist,
                    scaling_factor
                )
        )

        # Right Elbow
        scaledKeypoint.keypoints.right_elbow.x, scaledKeypoint.keypoints.right_elbow.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.left_shoulder,
                live_landmark.keypoints.right_shoulder,
                live_landmark.l_shoulder_r_shoulder_r_elbow,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.right_shoulder,
                    ref_landmark.keypoints.right_elbow,
                    scaling_factor
                )
            )

        # Right Wrist
        scaledKeypoint.keypoints.right_wrist.x, scaledKeypoint.keypoints.right_wrist.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.right_shoulder,
                live_landmark.keypoints.right_elbow,
                live_landmark.r_shoulder_r_elbow_r_wrist,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.right_elbow,
                    ref_landmark.keypoints.right_wrist,
                    scaling_factor
                )
            )

        # Left Knee
        scaledKeypoint.keypoints.left_knee.x, scaledKeypoint.keypoints.left_knee.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.right_hip,
                live_landmark.keypoints.left_hip,
                live_landmark.r_hip_l_hip_l_knee,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.left_hip,
                    ref_landmark.keypoints.left_knee,
                    scaling_factor
                )
            )
        
        # Left Ankle
        scaledKeypoint.keypoints.left_ankle.x, scaledKeypoint.keypoints.left_ankle.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.left_hip,
                live_landmark.keypoints.left_knee,
                live_landmark.l_hip_l_knee_l_ankle,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.left_knee,
                    ref_landmark.keypoints.left_ankle,
                    scaling_factor
                )
            )
        
        # Right Knee
        scaledKeypoint.keypoints.right_knee.x, scaledKeypoint.keypoints.right_knee.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.left_hip,
                live_landmark.keypoints.right_hip,
                live_landmark.l_hip_r_hip_r_knee,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.right_hip,
                    ref_landmark.keypoints.right_knee,
                    scaling_factor
                )
            )
        
        # Right Ankle
        scaledKeypoint.keypoints.right_ankle.x, scaledKeypoint.keypoints.right_ankle.y = KeypointScaling.compute_scaled_new_keypoint(
                live_landmark.keypoints.right_hip,
                live_landmark.keypoints.right_knee,
                live_landmark.r_hip_r_knee_r_ankle,
                KeypointScaling.scale_keypoint_distance(
                    ref_landmark.keypoints.right_knee,
                    ref_landmark.keypoints.right_ankle,
                    scaling_factor
                )
            )
        
        return scaledKeypoint
