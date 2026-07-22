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


# ---------------------------------
# 스트링 확률 계산
# ---------------------------------
def calculate_string_probability(px, py, string_lines):
    if len(string_lines) == 0:
        return []

    distances = []
    for (x1, y1, x2, y2) in string_lines:
        d = point_to_line_distance(px, py, x1, y1, x2, y2)
        distances.append(d)

    inv = [1.0 / (d + 1e-6) for d in distances]
    total = sum(inv)
    probs = [v / total for v in inv]
    return probs


# ---------------------------------
# 영상 열기
# ---------------------------------
video_path = "test_video6.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 열기 실패:", video_path)
    exit()
else:
    print("영상 열기 성공")

cv2.namedWindow("Right Hand Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Hand Test", 1280, 720)


# ---------------------------------
# 메인 루프
# ---------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # ---------------------------------
    # 스트링 검출
    # ---------------------------------
    data = detect_strings(frame)
    string_lines = data["string_lines"]

    for (x1, y1, x2, y2) in string_lines:
        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------------------------
    # YOLO 오른손 탐지
    # ---------------------------------
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

                for hand_landmarks in mp_result.multi_hand_landmarks:

                    h, w, _ = roi.shape

                    # 사용할 손가락 팁들
                    # 사용할 손가락 팁들 (엄지 포함)
                    finger_tips = {
                        "Thumb": 4,
                        "Index": 8,
                        "Middle": 12,
                        "Ring": 16,
                        "Pinky": 20
                    }

                    best_finger = None
                    best_prob_value = -1
                    best_tip_pos = None
                    best_probs = None

                    for finger_name, lm_id in finger_tips.items():

                        tip = hand_landmarks.landmark[lm_id]

                        tip_x = int(tip.x * w) + x1
                        tip_y = int(tip.y * h) + y1

                        probs = calculate_string_probability(
                            tip_x, tip_y, string_lines
                        )

                        if len(probs) == 0:
                            continue

                        max_prob = max(probs)

                        if max_prob > best_prob_value:
                            best_prob_value = max_prob
                            best_finger = finger_name
                            best_tip_pos = (tip_x, tip_y)
                            best_probs = probs

                    # ---------------------------------
                    # 가장 유력한 손가락만 표시
                    # ---------------------------------
                    if best_tip_pos is not None:

                        cv2.circle(display, best_tip_pos, 8, (0, 0, 255), -1)

                        cv2.putText(
                            display,
                            f"Finger: {best_finger}",
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        # 스트링 확률 표시
                        for i, p in enumerate(best_probs):
                            cv2.putText(
                                display,
                                f"S{i+1}: {p:.2f}",
                                (30, 70 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2
                            )

    cv2.imshow("Right Hand Test", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("프로그램 종료")