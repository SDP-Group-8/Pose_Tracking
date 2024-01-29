import mediapipe as mp
import cv2
from helper_funcs import *
import numpy as np


# Load the pose model
poseModule = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = poseModule.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    model_complexity=0
)

# ---------------------------------
# Helper FUNCTIONS
# ---------------------------------

#Extract key landmarks from pose results for further processing
# Returns a dictionary of key landmarks
def getLandmarks(poseResults):
    LEFT_SHOULDER = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_SHOULDER]
    RIGHT_SHOULDER = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_SHOULDER]
    LEFT_ELBOW = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_ELBOW]
    RIGHT_ELBOW = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_ELBOW]
    LEFT_WRIST = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_WRIST]
    RIGHT_WRIST = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_WRIST]

    LEFT_HIP = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_HIP]
    RIGHT_HIP = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_HIP]
    LEFT_KNEE = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_KNEE]
    RIGHT_KNEE = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_KNEE]
    LEFT_ANKLE = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_ANKLE]
    RIGHT_ANKLE = poseResults.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_ANKLE]

    final = {
        "LEFT_SHOULDER": LEFT_SHOULDER,
        "RIGHT_SHOULDER": RIGHT_SHOULDER,
        "LEFT_ELBOW": LEFT_ELBOW,
        "RIGHT_ELBOW": RIGHT_ELBOW,
        "LEFT_WRIST": LEFT_WRIST,
        "RIGHT_WRIST": RIGHT_WRIST,
        "LEFT_HIP": LEFT_HIP,
        "RIGHT_HIP": RIGHT_HIP,
        "LEFT_KNEE": LEFT_KNEE,
        "RIGHT_KNEE": RIGHT_KNEE,
        "LEFT_ANKLE": LEFT_ANKLE,
        "RIGHT_ANKLE": RIGHT_ANKLE
    }
    return final

#Calculate and return angles between various key landmarks
def getAngleData(landmarks):
    RIGHT_SHOULDER = landmarks['RIGHT_SHOULDER']
    LEFT_SHOULDER = landmarks['LEFT_SHOULDER']
    LEFT_ELBOW = landmarks['LEFT_ELBOW']
    LEFT_WRIST = landmarks['LEFT_WRIST']
    RIGHT_ELBOW = landmarks['RIGHT_ELBOW']
    RIGHT_WRIST = landmarks['RIGHT_WRIST']
    RIGHT_HIP = landmarks['RIGHT_HIP']
    LEFT_HIP = landmarks['LEFT_HIP']
    LEFT_KNEE = landmarks['LEFT_KNEE']
    LEFT_ANKLE = landmarks['LEFT_ANKLE']
    RIGHT_KNEE = landmarks['RIGHT_KNEE']
    RIGHT_ANKLE = landmarks['RIGHT_ANKLE']
    
    anglesData = [
        calculateAngle(RIGHT_SHOULDER, LEFT_SHOULDER, LEFT_ELBOW),
        calculateAngle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        calculateAngle(LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW),
        calculateAngle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        calculateAngle(RIGHT_HIP, LEFT_HIP, LEFT_KNEE),
        calculateAngle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        calculateAngle(LEFT_HIP, RIGHT_HIP, RIGHT_KNEE),
        calculateAngle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    ]
    return anglesData

# Adjusts the pose landmarks of pose2 to match the angles and scale of pose1
def interpolate(pose1Landmarks, pose2Landmarks):
    # Extract angle data from pose1
    angles_data = getAngleData(pose1Landmarks)  # This function calculates angles between specified landmarks in pose1

    # Initialize landmarks for pose2
    LEFT_SHOULDER = pose2Landmarks["LEFT_SHOULDER"]
    RIGHT_SHOULDER = pose2Landmarks["RIGHT_SHOULDER"]
    LEFT_ELBOW = pose2Landmarks["LEFT_ELBOW"]
    RIGHT_ELBOW = pose2Landmarks["RIGHT_ELBOW"]
    LEFT_WRIST = pose2Landmarks["LEFT_WRIST"]
    RIGHT_WRIST = pose2Landmarks["RIGHT_WRIST"]
    LEFT_HIP = pose2Landmarks["LEFT_HIP"]
    RIGHT_HIP = pose2Landmarks["RIGHT_HIP"]
    LEFT_KNEE = pose2Landmarks["LEFT_KNEE"]
    RIGHT_KNEE = pose2Landmarks["RIGHT_KNEE"]
    LEFT_ANKLE = pose2Landmarks["LEFT_ANKLE"]
    RIGHT_ANKLE = pose2Landmarks["RIGHT_ANKLE"]

    # Initialize corresponding landmarks for pose1
    LEFT_SHOULDER1 = pose1Landmarks["LEFT_SHOULDER"]
    RIGHT_SHOULDER1 = pose1Landmarks["RIGHT_SHOULDER"]
    LEFT_ELBOW1 = pose1Landmarks["LEFT_ELBOW"]
    RIGHT_ELBOW1 = pose1Landmarks["RIGHT_ELBOW"]
    LEFT_WRIST1 = pose1Landmarks["LEFT_WRIST"]
    RIGHT_WRIST1 = pose1Landmarks["RIGHT_WRIST"]
    LEFT_HIP1 = pose1Landmarks["LEFT_HIP"]
    RIGHT_HIP1 = pose1Landmarks["RIGHT_HIP"]
    LEFT_KNEE1 = pose1Landmarks["LEFT_KNEE"]
    RIGHT_KNEE1 = pose1Landmarks["RIGHT_KNEE"]
    LEFT_ANKLE1 = pose1Landmarks["LEFT_ANKLE"]
    RIGHT_ANKLE1 = pose1Landmarks["RIGHT_ANKLE"]

    # Calculate scaling factor based on shoulder distance
    scaling_factor = calculateDistance(LEFT_SHOULDER, RIGHT_SHOULDER)/calculateDistance(LEFT_SHOULDER1, RIGHT_SHOULDER1)

    # Process each limb of pose2 to match the corresponding limb in pose1
    # The process involves adjusting the position of each limb in pose2 based on the angles and distances in pose1

    # Adjust LEFT_ELBOW position
    angle = angles_data[0]
    dist = calculateDistance(LEFT_SHOULDER1, LEFT_ELBOW1) * scaling_factor
    x_coord, y_coord = computePoint(RIGHT_SHOULDER.x, RIGHT_SHOULDER.y, LEFT_SHOULDER.x, LEFT_SHOULDER.y, angle, dist)
    LEFT_ELBOW.x = x_coord
    LEFT_ELBOW.y = y_coord

    # Adjust LEFT_WRIST position
    angle = angles_data[1]
    dist = calculateDistance(LEFT_ELBOW1, LEFT_WRIST1) * scaling_factor
    x_coord, y_coord = computePoint(LEFT_SHOULDER.x, LEFT_SHOULDER.y, LEFT_ELBOW.x, LEFT_ELBOW.y, angle, dist)
    LEFT_WRIST.x = x_coord
    LEFT_WRIST.y = y_coord

    # Adjust RIGHT_ELBOW position
    angle = angles_data[2]
    dist = calculateDistance(RIGHT_SHOULDER1, RIGHT_ELBOW1) * scaling_factor
    x_coord, y_coord = computePoint(LEFT_SHOULDER.x, LEFT_SHOULDER.y, RIGHT_SHOULDER.x, RIGHT_SHOULDER.y, angle, dist)

    RIGHT_ELBOW.x = x_coord
    RIGHT_ELBOW.y = y_coord

    # Adjust RIGHT_WRIST position
    angle = angles_data[3]
    dist = calculateDistance(RIGHT_ELBOW1, RIGHT_WRIST1) * scaling_factor
    x_coord, y_coord = computePoint(RIGHT_SHOULDER.x, RIGHT_SHOULDER.y, RIGHT_ELBOW.x, RIGHT_ELBOW.y, angle, dist)

    RIGHT_WRIST.x = x_coord
    RIGHT_WRIST.y = y_coord

    # Adjust LEFT_KNEE position
    angle = angles_data[4]
    dist = calculateDistance(LEFT_HIP1, LEFT_KNEE1) * scaling_factor
    x_coord, y_coord = computePoint(RIGHT_HIP.x, RIGHT_HIP.y, LEFT_HIP.x, LEFT_HIP.y, angle, dist)

    LEFT_KNEE.x = x_coord
    LEFT_KNEE.y = y_coord

    # Adjust LEFT_ANKLE position
    angle = angles_data[5]
    dist = calculateDistance(LEFT_KNEE1, LEFT_ANKLE1) * scaling_factor
    x_coord, y_coord = computePoint(LEFT_HIP.x, LEFT_HIP.y, LEFT_KNEE.x, LEFT_KNEE.y, angle, dist)

    LEFT_ANKLE.x = x_coord
    LEFT_ANKLE.y = y_coord

    # Adjust RIGHT_KNEE position
    angle = angles_data[6]
    dist = calculateDistance(RIGHT_HIP1, RIGHT_KNEE1)* scaling_factor
    x_coord, y_coord = computePoint(LEFT_HIP.x, LEFT_HIP.y, RIGHT_HIP.x, RIGHT_HIP.y, angle, dist)

    RIGHT_KNEE.x = x_coord
    RIGHT_KNEE.y = y_coord

    # Adjust RIGHT_ANKLE position
    angle = angles_data[7]
    dist = calculateDistance(RIGHT_KNEE1, RIGHT_ANKLE1)* scaling_factor
    x_coord, y_coord = computePoint(RIGHT_HIP.x, RIGHT_HIP.y, RIGHT_KNEE.x, RIGHT_KNEE.y, angle, dist)

    RIGHT_ANKLE.x = x_coord
    RIGHT_ANKLE.y = y_coord

    # Return the updated pose2 with adjusted landmarks
    return {
    "LEFT_SHOULDER": LEFT_SHOULDER,
    "RIGHT_SHOULDER": RIGHT_SHOULDER,
    "LEFT_ELBOW": LEFT_ELBOW,
    "RIGHT_ELBOW": RIGHT_ELBOW,
    "LEFT_WRIST": LEFT_WRIST,
    "RIGHT_WRIST": RIGHT_WRIST,
    "LEFT_HIP": LEFT_HIP,
    "RIGHT_HIP": RIGHT_HIP,
    "LEFT_KNEE": LEFT_KNEE,
    "RIGHT_KNEE": RIGHT_KNEE,
    "LEFT_ANKLE": LEFT_ANKLE,
    "RIGHT_ANKLE": RIGHT_ANKLE
    }

def compareLandmarks(user_landmarks, ref_landmarks, landmark_weights=None):
    """
    Compares two sets of landmarks and returns a similarity score based on the 2D Euclidean distances.

    TODO: add support for Dynamic Time Warping (DTW) to account for temporal differences.
    TODO: try to find a way to segment set of landmarks into distinct moves

    Args:
    user_landmarks (dict): Landmarks from the user's dance frame.
    ref_landmarks (dict): Corresponding landmarks from the reference dance frame.
    landmark_weights (dict, optional): A dictionary containing weights for each landmark.

    Returns:
    float: A score representing the similarity between the two sets of landmarks. Lower scores indicate higher similarity.
    """
    score = 0
    total_weight = 0

    # Default weights (equal importance to all landmarks)
    if landmark_weights is None:
        landmark_weights = {landmark: 1 for landmark in user_landmarks.keys()}

    user_landmarks_scaled = interpolate(ref_landmarks, user_landmarks)

    # Iterate over the landmarks and calculate weighted distances
    for landmark in user_landmarks_scaled.keys():
        if landmark in ref_landmarks:
            # for each landmark, calculate the distance between the user's pose and the reference pose
            distance = calculateDistance(user_landmarks_scaled[landmark], ref_landmarks[landmark])
            weight = landmark_weights.get(landmark, 1)

            # Add the weighted distance to the score
            score += distance * weight
            total_weight += weight

    # Normalize the score by the total weight
    if total_weight > 0:
        score /= total_weight

    return score

def adaptive_segmented_dtw(a, b, num_segments=10):
    """
    Performs DTW on adaptively sized segments of two sequences of possibly different lengths.

    Args:
    a, b (list): Sequences of data points.
    num_segments (int): Number of segments to divide the sequences into.

    Returns:
    np.array: Array of DTW distances for each segment.
    """
    final_distances = []
    len_a, len_b = len(a), len(b)
    
    # Calculate segment sizes for each sequence
    segment_size_a = len_a // num_segments
    segment_size_b = len_b // num_segments

    for segment in range(num_segments):
        start_a = segment * segment_size_a
        end_a = start_a + segment_size_a
        start_b = segment * segment_size_b
        end_b = start_b + segment_size_b

        # Handle last segment to include any remaining elements due to integer division
        if segment == num_segments - 1:
            end_a = len_a
            end_b = len_b

        # Extract segments
        segment_a = a[start_a:end_a]
        segment_b = b[start_b:end_b]

        # Apply DTW to the segments
        distance, _ = fastdtw(segment_a, segment_b, dist=euclidean)
        final_distances.append(distance)

    return np.array(final_distances)


# ---------------------------------
# POSE ESTIMATOR FUNCTIONS
# ---------------------------------

# Processes a video to extract frame by frame pose landmarks
# Returns a list of pose landmarks for each frame
def processVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    rawFramePose = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose landmarks
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Add the KEY landmarks to the list
            rawFramePose.append(results)
            
    cap.release()
    frameKeyLandmarks = [getLandmarks(results) for results in rawFramePose]
    return frameKeyLandmarks, rawFramePose

def processLiveVideo(camera_id, ref_landmarks, max_frames=None, display_live=False, raw_ref_pose=None):
    """
    Processes a live video feed from a camera and compares it with a set of reference dance landmarks.

    Args:
    camera_id (int): The ID of the camera to be used for capturing live video.
    ref_landmarks (list): A list of pose landmarks extracted from the reference dance video.
    max_frames (int, optional): The maximum number of frames to process from the live feed.
    display_live (bool, optional): Flag to control whether the live feed is displayed on-screen.

    Returns:
    float: overall similarity score between the live and reference videos. 
    """

    # Initialize the camera for live video capture
    cap_live = cv2.VideoCapture(camera_id)

    frame_counter = 0   # Counter to track the number of frames processed
    total_score = 0     # Variable to accumulate the total score
    frame_scores = []   # List to store scores for each frame

    # Process the video frame by frame
    while cap_live.isOpened() and (max_frames is None or frame_counter < max_frames):
        ret, frame = cap_live.read()  # Read a frame from the camera
        if not ret:  # Break the loop if no frame is captured
            break

        # Convert the frame to RGB as MediaPipe requires RGB images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe to get pose landmarks
        results = pose.process(frame_rgb)

        # Check if landmarks are detected in the frame
        if results.pose_landmarks:
            # If display is enabled, show the live feed with the current frame score
            if display_live:
                mp_drawing.draw_landmarks(frame, raw_ref_pose[frame_counter].pose_landmarks, poseModule.POSE_CONNECTIONS, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, poseModule.POSE_CONNECTIONS, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 225)))
                cv2.imshow("Live Dance Grading", frame)
                # Check for 'ESC' key press to exit
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Extract landmarks from the current frame
            user_landmarks = getLandmarks(results)

            # Get corresponding landmarks from the reference video
            # Uses modulo operation to loop over reference landmarks if needed
            ref_frame_landmarks = ref_landmarks[frame_counter % len(ref_landmarks)]

            # Compare current frame's landmarks with reference frame's landmarks
            frame_score = compareLandmarks(user_landmarks, ref_frame_landmarks)

            # Store and accumulate the score
            frame_scores.append(frame_score)
            total_score += frame_score

            # Increment the frame counter
            frame_counter += 1

    # Release the camera resource and close any open windows
    cap_live.release()
    cv2.destroyAllWindows()

    # Calculate the overall score by averaging the frame scores
    overall_score = total_score / frame_counter if frame_counter > 0 else 0

    return overall_score


video_path = "dances/renegade.mp4"
ref_landmarks, raw_ref_pose = processVideo(video_path)

print(processLiveVideo(0, ref_landmarks, len(ref_landmarks), display_live=True, raw_ref_pose=raw_ref_pose))