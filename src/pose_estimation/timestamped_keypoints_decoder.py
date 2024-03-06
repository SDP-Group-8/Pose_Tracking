from pose_estimation.keypoint_decoder import KeypointDecoder

import json

class TimestampedKeypointsDecoder(json.JSONDecoder):
    timestamp_field_name = "timestamp_ms"
    keypoints_field_name = "keypoints"

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, repr: str):
        decoder = KeypointDecoder()
        keypoints = json.loads(repr[TimestampedKeypointsDecoder.keypoints_field_name])
        return repr[TimestampedKeypointsDecoder.timestamp_field_name], decoder.object_hook(keypoints)
