import argparse

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from pose_estimation.mediapipe import MediaPipe
from pose_estimation.capture_device import CaptureDevice

class SingleWindow:
    window_name = "pose_detections"

    def __init__(self, capture_device: CaptureDevice = None, image: mp.Image = None):
        '''
        Initialize window object with either image or video data source
        :param capture_device: video data source
        :param image: image data source, using custom mediapipe format
        '''
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        width = capture_device.get_width() if capture_device else image.width
        height = capture_device.get_height() if capture_device else image.height
        cv2.resizeWindow("Ouput", int(width), int(height))

    def draw_and_show(self, image: np.ndarray, detections: [NormalizedLandmark]):
        '''
        Draw pose detections on the input image and display them
        :param image: image
        :param detections: Pose detections on the image
        '''
        annotated_image = SingleWindow.draw_pose_on_image(image, detections)
        cv2.imshow(self.window_name, annotated_image)
        cv2.waitKey(10)

    @staticmethod
    def draw_pose_on_image(image: np.ndarray, detections: [NormalizedLandmark]) -> np.ndarray:
        '''
        Draw pose detections on the input
        :param image: image
        :param detections: Pose detections on the image
        :return: image with pose detections visualized
        '''
        normalized_landmarks = [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in detections]
        annotated_image = np.copy(image)

        # Draw the pose landmarks.
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks.landmark.extend(normalized_landmarks)

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        
        return annotated_image
    
    def destroy(self):
        cv2.destroyWindow(self.window_name)

    def should_close(self):
        if cv2.waitKey(25) & 0xFF == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True
        return False


def estimate_image():
    parser = argparse.ArgumentParser()

    parser.add_argument("file")
    args = parser.parse_args()

    image = mp.Image.create_from_file(args.file)
    window = SingleWindow(image=image)

    media_pipe = MediaPipe()
    media_pipe.initialize(mode=vision.RunningMode.IMAGE)

    res = media_pipe.process_image(image)
    window.draw_and_show(image.numpy_view(), res)


def estimate_video():
    parser = argparse.ArgumentParser()

    parser.add_argument("file")
    parser.add_argument("-live", action="store_true", dest="live")
    args = parser.parse_args()

    media_pipe = MediaPipe()
    media_pipe.initialize()

    cap = CaptureDevice(args.file, live=args.live)
    window = SingleWindow(cap)

    while cap.is_opened():
        frame_exists, frame = cap.read()
        if frame_exists:
            timestamp = int(cap.get_timestamp().total_seconds() * 1e3)
            res = media_pipe.process_frame(frame, timestamp = timestamp)
            window.draw_and_show(frame, res)

        if window.should_close():
            break
    
    window.destroy()
    cap.close()