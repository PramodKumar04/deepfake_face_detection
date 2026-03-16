import cv2


def extract_frames(video_path: str, target_fps: int = 5):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25

    interval = max(int(original_fps // target_fps), 1)

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            frames.append(frame)

        frame_id += 1

    cap.release()
    return frames
