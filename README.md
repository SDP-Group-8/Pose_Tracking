# Pose_Tracking
First attempt at getting pose tracking setup and working

# Quick Startup

Warning: This works on mine currently but you might be missing prerequisites which I have on my machine.

1. Install the project locally
2. Get the model file

```
pip install --editable .
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

# Scripts

```
estimate-video <video_source> -live
estimate-image <image_path>
```

# Useful Links
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python