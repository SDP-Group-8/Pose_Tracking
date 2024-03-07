from __future__ import annotations

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import json

class KeypointDecoder:
    def default(self, repr: str):
        return NormalizedLandmark(**json.loads(repr))
