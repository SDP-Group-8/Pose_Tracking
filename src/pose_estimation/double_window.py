import argparse
import numpy as np
import cv2

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from pose_estimation.mediapipe import MediaPipe
from pose_estimation.mediapipe_video import MediaPipeVideo
from pose_estimation.capture_device import CaptureDevice
from pose_estimation.single_window import SingleWindow
from pose_estimation.pre_processing.keypoint_scaling import KeypointScaling
from pose_estimation.keypoint_statistics import KeypointStatistics

class DoubleWindow:
    window_name = "pose_detections"

    def __init__(self, capture_device: CaptureDevice = None, reference_video: CaptureDevice = None):
        '''
        Initialize window object with either image or video data source
        :param capture_device: video data source
        :param reference_video: reference data source
        '''
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        width = int(capture_device.get_width() + reference_video.get_width())
        height = max(capture_device.get_height(), reference_video.get_height())
        cv2.resizeWindow(self.window_name, int(width), int(height))

    def draw_and_show(self, image1: np.ndarray, detections1: [NormalizedLandmark], image2: np.ndarray, detections2: [NormalizedLandmark], score: float):
        '''
        Draw pose detections on the inputted images and display them
        :param image1: image1
        :param detections1: Pose detections of image1
        :param image2: image2
        :param detections2: Pose detections of image2
        '''
        # Handling when no detections are found
        if detections1:
            annotated_image1 = SingleWindow.draw_pose_on_image(image1, detections1)
        else:
            annotated_image1 = image1
        if detections2:
            annotated_image2 = SingleWindow.draw_pose_on_image(image2, detections2)
        else:
            annotated_image2 = image2

        annotated_image = cv2.hconcat([annotated_image1, annotated_image2])
        
        # Specify the text, font, and other parameters
        text = "Score: " + str(score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)  # White color in BGR
        position = (10, 50)  # Coordinates of the starting point of the text

        # Add text to the image
        cv2.putText(annotated_image, text, position, font, font_scale, font_color, font_thickness)

        cv2.imshow(self.window_name, annotated_image)
        cv2.waitKey(10)
    
    def destroy(self):
        cv2.destroyWindow(self.window_name)

    def should_close(self):
        if cv2.waitKey(25) & 0xFF == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True
        return False


def estimate_live_video_comparison():
    parser = argparse.ArgumentParser()

    parser.add_argument("cam_num")
    parser.add_argument("reference_video")
    args = parser.parse_args()

    media_pipe = MediaPipe()
    media_pipe.initialize()

    media_pipe_video = MediaPipeVideo(args.reference_video)
    ref_pose_data = media_pipe_video.estimate_video()

    live = CaptureDevice(args.cam_num, True)
    ref = CaptureDevice(args.reference_video, False)
    window = DoubleWindow(live, ref)
    
    frame_count = 0
    while ref.is_opened():
        ref_frame_exists, reference_frame = ref.read()
        live_frame_exists, live_frame = live.read()
        
        if live_frame_exists and ref_frame_exists:
            live_timestamp = int(live.get_timestamp())
            live_detections = media_pipe.process_frame(live_frame, timestamp = live_timestamp)

            reference_detections = ref_pose_data[frame_count]

            frame_count += 1

            reference_statistics = KeypointStatistics.from_keypoints(reference_detections)

            # Handling when no user is detected in the frame
            if (live_detections):

                live_statistics = KeypointStatistics.from_keypoints(live_detections)
                scaled_keypoints = KeypointScaling.scale_keypoints(reference_statistics, live_statistics)

                # TODO add scoring in here, once branch is merged
                score = 0

            else:
                print("User not in frame!")

            # Print original detections, don't want to print scaled ones
            window.draw_and_show(
                reference_frame, 
                reference_detections.to_normalized_landmarks() if reference_detections else None,
                live_frame,
                live_detections.to_normalized_landmarks() if live_detections else None, 
                score
            )

        if window.should_close():
            break
    
    window.destroy()
    ref.close()
    live.close()
