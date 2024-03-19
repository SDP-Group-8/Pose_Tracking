from pose_estimation.keypoints import Keypoints

class Interpolator:
    def __init__(self, prev_keypoints: Keypoints, next_keypoints: Keypoints):
        self.prev_keypoints = prev_keypoints
        self.next_keypoints = next_keypoints

    def weighted_average(self, prev_timestamp_ms, timestamp_ms, next_timestamp_ms):
        if self.prev_keypoints is None:
            return self.next_keypoints
        elif self.next_keypoints is None:
            return self.prev_keypoints
        
        prev_weight = (timestamp_ms - prev_timestamp_ms) / (next_timestamp_ms - prev_timestamp_ms)
        next_weight = (next_timestamp_ms - timestamp_ms) / (next_timestamp_ms - prev_timestamp_ms)

        average = prev_weight * self.prev_keypoints.to_numpy_positions() + next_weight * self.next_keypoints.to_numpy_positions()
        return Keypoints.from_numpy_positions(average)
