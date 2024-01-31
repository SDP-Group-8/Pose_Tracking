import argparse
import math

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from pose_estimation.mediapipe import MediaPipe
from pose_estimation.capture_device import CaptureDevice

class DoubleWindow:
    window_name = "pose_detections"

    def __init__(self, capture_device: CaptureDevice = None, reference_video: CaptureDevice = None):
        '''
        Initialize window object with either image or video data source
        :param capture_device: video data source
        :param image: image data source, using custom mediapipe format
        '''
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        width = int(capture_device.get_width() + reference_video.get_width())
        height = max(capture_device.get_height(), reference_video.get_height())
        cv2.resizeWindow(self.window_name, int(width), int(height))

    def draw_and_show(self, image1: np.ndarray, detections1: [NormalizedLandmark], image2: np.ndarray, detections2: [NormalizedLandmark]):
        '''
        Draw pose detections on the inputted images and display them
        :param image1: image1
        :param detections1: Pose detections of image1
        :param image2: image2
        :param detections2: Pose detections of image2
        '''
        annotated_image1 = DoubleWindow.draw_pose_on_image(image1, detections1)
        annotated_image2 = DoubleWindow.draw_pose_on_image(image2, detections2)
        annotated_image = cv2.hconcat([annotated_image1, annotated_image2])
        cv2.imshow(self.window_name, annotated_image)
        cv2.waitKey(10)

    @staticmethod
    def draw_pose_on_image(image: np.ndarray, detections: [NormalizedLandmark]) -> np.ndarray:
        '''
        Draw pose detections on the inputted images
        :param image: image
        :param detections: Pose detections on the image
        :return: image with pose detections visualized
        '''
        if detections is None:
            return image
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


def estimateLiveVideoComparison():
    parser = argparse.ArgumentParser()

    parser.add_argument("cam_num")
    parser.add_argument("reference_video")
    args = parser.parse_args()

    media_pipe = MediaPipe()
    media_pipe.initialize()

    refPoseData = estimateVideo(args.reference_video)

    live = CaptureDevice(args.cam_num, True)
    ref = CaptureDevice(args.reference_video, False)
    window = DoubleWindow(live, ref)
    
    frameCount = 0
    while ref.is_opened():
        ref_frame_exists, refFrame = ref.read()
        live_frame_exists, liveFrame = live.read()
        
        if live_frame_exists and ref_frame_exists:
            liveTimestamp = int(live.get_timestamp())
            liveRes = media_pipe.process_frame(liveFrame, timestamp = liveTimestamp)

            refRes = refPoseData[frameCount]
            frameCount += 1

            window.draw_and_show(refFrame, refRes, liveFrame, liveRes)

        if window.should_close():
            break
    
    window.destroy()
    ref.close()
    live.close()

def estimateVideo(video):
    mp = MediaPipe()
    mp.initialize()

    video = CaptureDevice(video, False)
    results = []
    while video.is_opened():
        frame_exists, frame = video.read()
        if frame_exists:
            timestamp = int(video.get_timestamp())
            res = mp.process_frame(frame, timestamp)
            results.append(res)
        else:
            break
    video.close()
    return results
