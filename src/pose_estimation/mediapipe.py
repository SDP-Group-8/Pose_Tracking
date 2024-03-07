from __future__ import annotations

import numpy as np
from mediapipe.tasks import python
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark_detection_result import LandmarksDetectionResult

from pose_estimation.keypoints import Keypoints

class MediaPipe:
    # different options of model to use "pose_landmarker_{lite/full/heavy}"
    # Download them from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
    def initialize(self, parameter_path: str, mode=vision.RunningMode.VIDEO):
        '''
        Initialize the mediapipe detection backend
        :param parameter_path: path to the model description / parameter file
        :param mode: running mode for mediapipe (likely either image or video)
        '''
        base_options = python.BaseOptions(model_asset_path=parameter_path)

        # Set Pose Estimator Options
        options = vision.PoseLandmarkerOptions(
            base_options = base_options,
            output_segmentation_masks = True,
            running_mode= mode)

        # Create Landmarker
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame, timestamp: float) -> Keypoints:
        '''
        Convert the input video frame into the right format and detect landmarks
        :param frame: input frame from the stream
        :param timestamp: timestamp from the stream starting
        :return: normalized pose detections
        '''
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        return self.__extract_keypoints(self.landmarker.detect_for_video(image, timestamp))

    def process_image(self, image: np.ndarray) -> Keypoints:
        '''
        Detect keypoints on the (static) input image
        :param image: in numpy format
        :return: normalized pose detections
        '''
        return self.__extract_keypoints(self.landmarker.detect(image))
    
    def __extract_keypoints(self, landmarks: LandmarksDetectionResult) -> Keypoints | None:
        # If landmarks are not found (user out of frame) return None
        if len(landmarks.pose_landmarks) == 0:
            return None
        else:
            return Keypoints.from_normalized_landmarks(landmarks.pose_landmarks[0])