from __future__ import annotations

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import json

from pose_estimation.keypoints import Keypoints

class KeypointDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, repr):
        return Keypoints.from_dict({f"{name}": NormalizedLandmark(**json.loads(repr[name])) for name in Keypoints.ordered_fields})
