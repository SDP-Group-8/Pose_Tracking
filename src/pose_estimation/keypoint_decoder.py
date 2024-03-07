from __future__ import annotations

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import json

class KeypointDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, repr):
        return [NormalizedLandmark(**json.loads(props)) for props in (
            repr["left_shoulder"],
            repr["right_shoulder"],
            repr["left_elbow"],
            repr["right_elbow"],
            repr["left_wrist"],
            repr["right_wrist"],

            repr["left_hip"],
            repr["right_hip"],
            repr["left_knee"],
            repr["right_knee"],
            repr["left_ankle"],
            repr["right_ankle"]
        )]
