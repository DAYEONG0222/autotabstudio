import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # MediaPipe 로그 억제

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from src.new3_grid import detect_strings


# ---------------------------------
# 🔥 YOLO 오른손 모델 로드
# ---------------------------------
yolo = YOLO("weights_hands.pt")  # 손 모델 파일명 확인


# ---------------------------------
# 🔥 MediaPipe 설정
# ---------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# ---------------------------------
# 🔥 선분 거리 계산
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
# 🔥 스트링 확률 계산
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
# 🔥 영상 열기
# ---------------------------------
video_path = "test_video6.mp4"  # 영상 파일명 확인
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 영상 열기 실패:", video_path)
    exit()
else:
    print("✅ 영상 열기 성공")

# ---------------------------------
# 🔥 창 강제 생성 (전체화면 제거)
# ---------------------------------
cv2.namedWindow("Right Hand Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Hand Test", 1280, 720)


# ---------------------------------
# 🔥 메인 루프
# ---------------------------------
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 끝 또는 프레임 읽기 실패")
        break

    frame_count += 1
    display = frame.copy()

    # ---------------------------------
    # 1️⃣ 스트링 검출
    # ---------------------------------
    data = detect_strings(frame)
    string_lines = data["string_lines"]

    for (x1, y1, x2, y2) in string_lines:
        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------------------------
    # 2️⃣ YOLO 오른손 탐지 (속도 개선)
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
                    tip = hand_landmarks.landmark[8]  # 검지

                    tip_x = int(tip.x * w) + x1
                    tip_y = int(tip.y * h) + y1

                    cv2.circle(display, (tip_x, tip_y), 8, (0, 0, 255), -1)

                    # ---------------------------------
                    # 3️⃣ 스트링 확률 계산
                    # ---------------------------------
                    probs = calculate_string_probability(
                        tip_x, tip_y, string_lines
                    )

                    for i, p in enumerate(probs):
                        cv2.putText(
                            display,
                            f"S{i+1}: {p:.2f}",
                            (30, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2
                        )

    cv2.imshow("Right Hand Test", display)

    # ESC 키 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
print("프로그램 종료")