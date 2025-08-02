import cv2
import numpy as np
import mediapipe as mp

# Define cheek landmarks (customizable)
LEFT_CHEEK_LANDMARKS = [347, 348, 266, 425, 280]
RIGHT_CHEEK_LANDMARKS = [118, 119, 36, 205, 50]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def extract_cheek_patch(frame, landmarks, width=32, height=32):
    """Extract a cheek patch from a frame using facial landmarks."""
    h, w, _ = frame.shape
    coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    cheek_patch = frame[y_min:y_max, x_min:x_max]
    if cheek_patch.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)  # fallback
    return cv2.resize(cheek_patch, (width, height))

def process_video(video_path, width=32, height=32):
    """Process a video and extract left and right cheek patch sequences."""
    cap = cv2.VideoCapture(video_path)
    lc_imgs, rc_imgs = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_patch = extract_cheek_patch(rgb_frame, [landmarks[i] for i in LEFT_CHEEK_LANDMARKS], width, height)
            right_patch = extract_cheek_patch(rgb_frame, [landmarks[i] for i in RIGHT_CHEEK_LANDMARKS], width, height)
        else:
            left_patch = np.zeros((height, width, 3), dtype=np.uint8)
            right_patch = np.zeros((height, width, 3), dtype=np.uint8)

        lc_imgs.append(left_patch)
        rc_imgs.append(right_patch)

    cap.release()
    return np.array(lc_imgs, dtype=np.uint8), np.array(rc_imgs, dtype=np.uint8)
