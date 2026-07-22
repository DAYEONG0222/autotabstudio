import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from src.new3_grid import detect_strings


# ---------------------------------
# YOLO 모델 로드
# ---------------------------------
yolo = YOLO("weights_hands.pt")


# ---------------------------------
# MediaPipe 설정
# ---------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# ---------------------------------
# 선분 거리 계산
# ---------------------------------
def point_to_line_distance(px, py, x1, y1, x2, y2):
    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])

    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)

    t = np.dot(point_vec, line_vec) / line_len
    t = max(0, min(1, t))

    proj = np.array([x1, y1]) + t * line_vec
    return np.linalg.norm(np.array([px, py]) - proj)


def calculate_string_probability(px, py, string_lines):
    if len(string_lines) == 0:
        return []

    distances = []
    for (x1, y1, x2, y2) in string_lines:
        d = point_to_line_distance(px, py, x1, y1, x2, y2)
        distances.append(d)

    inv = [1.0 / (d + 1e-6) for d in distances]
    total = sum(inv)
    return [v / total for v in inv]


# ---------------------------------
# 영상 열기
# ---------------------------------
cap = cv2.VideoCapture("test_video6.mp4")

window_name = "Right Hand Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

prev_gray = None


# ---------------------------------
# 메인 루프
# ---------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    data = detect_strings(frame)
    string_lines = data["string_lines"]

    for (x1, y1, x2, y2) in string_lines:
        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    results = yolo(frame, imgsz=640, verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:

            cls_name = yolo.names[int(box.cls[0])]
            if cls_name != "right_hand":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_result = hands.process(rgb)

            if mp_result.multi_hand_landmarks:

                finger_tips = {
                    "Thumb": 4,
                    "Index": 8,
                    "Middle": 12,
                    "Ring": 16
                }

                best_finger = None
                best_tip_pos = None
                best_string = None
                best_score = 0

                for hand_landmarks in mp_result.multi_hand_landmarks:
                    h, w, _ = roi.shape

                    for name, lm_id in finger_tips.items():

                        tip = hand_landmarks.landmark[lm_id]
                        tip_x = int(tip.x * w) + x1
                        tip_y = int(tip.y * h) + y1

                        string_probs = calculate_string_probability(
                            tip_x, tip_y, string_lines
                        )

                        if len(string_probs) == 0:
                            continue

                        max_string_prob = max(string_probs)
                        string_idx = np.argmax(string_probs)

                        # tip 주변 motion 계산
                        if prev_gray is not None:
                            patch_size = 15
                            x_min = max(tip_x - patch_size, 0)
                            y_min = max(tip_y - patch_size, 0)
                            x_max = min(tip_x + patch_size, frame.shape[1])
                            y_max = min(tip_y + patch_size, frame.shape[0])

                            prev_patch = prev_gray[y_min:y_max, x_min:x_max]
                            curr_patch = gray[y_min:y_max, x_min:x_max]

                            if prev_patch.shape == curr_patch.shape:
                                diff = cv2.absdiff(prev_patch, curr_patch)
                                motion_score = np.mean(diff) / 255.0
                            else:
                                motion_score = 0
                        else:
                            motion_score = 0

                        finger_score = max_string_prob * motion_score

                        if finger_score > best_score:
                            best_score = finger_score
                            best_finger = name
                            best_tip_pos = (tip_x, tip_y)
                            best_string = string_idx

                if best_tip_pos is not None and best_score > 0.02:

                    cv2.circle(display, best_tip_pos, 10, (0, 0, 255), -1)

                    cv2.putText(display,
                                f"{best_finger} -> String {best_string+1}",
                                (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2)

    prev_gray = gray.copy()

    cv2.imshow(window_name, display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()