from __future__ import annotations

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import json

class KeypointEncoder(json.JSONEncoder):
    def default(self, o: NormalizedLandmark):
        return json.dumps({
            "x": o.x,
            "y": o.y,
            "z": o.z,
            "visibility": o.visibility,
            "presence": o.presence
        })
