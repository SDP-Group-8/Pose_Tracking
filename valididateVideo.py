from validateVideoExceptions import LowWidthException, LowHeightException, LowFPSException
import cv2


def validate_video(video_name: str) -> None:
    MINIMUM_WIDTH = 720
    MINIMUM_HEIGHT = 720
    TARGET_FPS = 30

    cap = cv2.VideoCapture(video_name)

    # Get video details
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = round(cap.get(cv2.CAP_PROP_FPS))

    # Checks if the input video is a 720x720 30fps video, as videos may be different aspect ratios, this ensures
    # the smallest point of the rectangle is at least 720 pixels.

    if width < MINIMUM_WIDTH:
        raise LowWidthException

    if height < MINIMUM_HEIGHT:
        raise LowHeightException

    if fps_original < TARGET_FPS:
        raise LowFPSException

    # How many times larger is the fps of the video compared to the target
    fps_multiplier = int(round(fps_original / TARGET_FPS))

    # Create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{video_name}_validated.mp4", fourcc, TARGET_FPS, (width, height))

    # Process and write frames
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to achieve the target fps
        if frame_number % fps_multiplier == 0:
            out.write(frame)

        frame_number += 1

    # End CV2 windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


validate_video("Add_Video_Name_Here.mp4")
