import sys
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
# DEFINE NEEDED FUNCTIONS
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

def getLandmarks(results):
    LEFT_SHOULDER = results.pose_landmarks[0][11]
    RIGHT_SHOULDER = results.pose_landmarks[0][12]
    LEFT_ELBOW = results.pose_landmarks[0][13]
    RIGHT_ELBOW = results.pose_landmarks[0][14]
    LEFT_WRIST = results.pose_landmarks[0][15]
    RIGHT_WRIST = results.pose_landmarks[0][16]

    LEFT_HIP = results.pose_landmarks[0][23]
    RIGHT_HIP = results.pose_landmarks[0][24]
    LEFT_KNEE = results.pose_landmarks[0][25]
    RIGHT_KNEE = results.pose_landmarks[0][26]
    LEFT_ANKLE = results.pose_landmarks[0][27]
    RIGHT_ANKLE = results.pose_landmarks[0][28]

    return [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

def getPoseDataFromVideo(video):
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(video)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Stores the pose data at each frame. with the index being the frame number.
    #videoPoseData = np.empty([frameCount, 12])
    videoPoseData = [[]] * (frameCount + 1)

    while cap.isOpened():   
        # Read frame from webcam video stream
        frameExists, frame = cap.read() 

        if frameExists:
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Send frame to Pose Landmarker for pose estimation.
            # Results accessible via the result_callback set in PoseLandmarkerOptions object called 'options'.
            timestamp = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            results = landmarker.detect_for_video(image, timestamp)

            

            videoPoseData[timestamp] = results
        else:
            break        
    cap.release()
    cv2.destroyAllWindows()
    return videoPoseData


def danceGame(referenceDance, camNum):
    referencePoseData = getPoseDataFromVideo(referenceDance)

    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    liveVideo = cv2.VideoCapture(camNum)
    refVideo = cv2.VideoCapture(referenceDance)

    # Check videos can be opened
    if not (liveVideo.isOpened() and refVideo.isOpened()):
        print("Error: reference video or live camera could not be accessed! :(")
        return

    # Create window for displaying results
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    windowWidth = liveVideo.get(3) + refVideo.get(3)
    windowHeight = max(liveVideo.get(4), refVideo.get(4))
    cv2.resizeWindow("Ouput", int(windowWidth), int(windowHeight))

    while refVideo.isOpened():   
        # Read frame from webcam video stream
        dancerFrameExists, dancerFrame = liveVideo.read()
        # Read frame from reference video
        refFrameExists, refFrame = refVideo.read()

        if dancerFrameExists and refFrameExists:
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            dancerImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=dancerFrame)
            refImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=refFrame)

            # Send frames to Pose Landmarker for pose estimation.
            # TODO HACK Figure out why this has to be multiplied by 1000 for it work?
            timestamp = int(refVideo.get(cv2.CAP_PROP_POS_FRAMES) * 1000)
            dancerResult = landmarker.detect_for_video(dancerImage, timestamp)

            # refTimestamp = int(refVideo.get(cv2.CAP_PROP_POS_FRAMES))
            # refResult = landmarker.detect_for_video(refImage, timestamp)
            
            # Draw the results on the images
            dancerResultImage = drawPoseOnImage(dancerImage.numpy_view(), dancerResult)
            refResultImage = drawPoseOnImage(refImage.numpy_view(), referencePoseData[int(timestamp/1000)])

            # Display the images
            combinedImage = cv2.hconcat([dancerResultImage, refResultImage])
            cv2.imshow("Output", combinedImage)
        else:
            break
        
        # Quit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        
    liveVideo.release()
    refVideo.release()
    cv2.destroyAllWindows()
    return

# -----
# MAIN
# -----
danceFilename = sys.argv[1]
danceGame(danceFilename, 0)


