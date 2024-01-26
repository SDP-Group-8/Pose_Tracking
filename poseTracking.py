import numpy as np
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# ----------------------
# CREATE POSE ESTIMATOR.
# -----------------------

# different options of model to use "pose_landmarker_{lite/full/heavy}"
# Download them from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
model_path = 'pose_landmarker_full.task'

# Get base options
base_options = python.BaseOptions(model_asset_path=model_path)

# Set Pose Estimator Options
options = vision.PoseLandmarkerOptions(
    base_options = base_options,
    output_segmentation_masks = True,
    running_mode= vision.RunningMode.VIDEO)

# Create Landmarker
landmarker = vision.PoseLandmarker.create_from_options(options)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

# Draws inputted pose onto an image
def drawPoseOnImage(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# ---------------------------------
# POSE ESTIMATOR FUNCTIONS
# ---------------------------------

def poseEstimationLiveStream():
    # When open CV is fed 0 as the video file, it uses the webcam as a live feed
    poseEstimationOfVideo(0)
    return

def poseEstimationOfVideo(video):
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(video)
    while cap.isOpened():   
        # Read frame from webcam video stream
        frameExists, frame = cap.read() 

        if frameExists:
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Send frame to Pose Landmarker for pose estimation.
            # Results accessible via the result_callback set in PoseLandmarkerOptions object called 'options'.
            # TODO How should timestamp be worked out?
            timestamp = int(time.time() * 1000)
            result = landmarker.detect_for_video(image, timestamp)
            
            # Draw the results on the image
            resultImage = drawPoseOnImage(image.numpy_view(), result)
            # Display the image
            cv2.imshow('Output', resultImage)
        else:
            break
        
        # Quit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return

# Uncomment whichever version you want to run
# poseEstimationOfVideo('{video name here}')
poseEstimationLiveStream()