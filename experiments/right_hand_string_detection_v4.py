import cv2
import mediapipe as mp
import numpy as np

# =============================
# 설정
# =============================
VIDEO_PATH = "test_video6.mp4"   # ← 분석할 영상 파일 경로
OUTPUT_PATH = "output.mp4"

STRING_COUNT = 6
STRING_AREA_RATIO = 0.5
VIBRATION_THRESHOLD = 8
STRIP_WIDTH = 6
FINGER_TIPS = [4, 8, 12, 16, 20]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상 파일 열기 실패")
    exit()

# 영상 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

prev_gray = None

# =============================
# 줄 생성
# =============================
def create_strings(w, h):
    strings = []
    y_start = int(h * (0.5 - STRING_AREA_RATIO/2))
    y_end   = int(h * (0.5 + STRING_AREA_RATIO/2))
    spacing = (y_end - y_start) // (STRING_COUNT - 1)

    for i in range(STRING_COUNT):
        y = y_start + i * spacing
        strings.append((0, y, w, y))
    return strings

# =============================
# strip 추출
# =============================
def extract_string_strip(gray_frame, line, width=6):
    mask = np.zeros_like(gray_frame)
    x1, y1, x2, y2 = line
    cv2.line(mask, (x1, y1), (x2, y2), 255, width)
    strip = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
    return strip, mask

# =============================
# 진동 감지
# =============================
def detect_string_vibration(prev_gray, curr_gray, string_lines):
    vibrating = []

    for idx, line in enumerate(string_lines):
        prev_strip, mask = extract_string_strip(prev_gray, line, STRIP_WIDTH)
        curr_strip, _ = extract_string_strip(curr_gray, line, STRIP_WIDTH)

        diff = cv2.absdiff(prev_strip, curr_strip)
        score = np.sum(diff) / (np.sum(mask > 0) + 1e-6)

        if score > VIBRATION_THRESHOLD:
            vibrating.append(idx)

    return vibrating

# =============================
# 확률 계산
# =============================
def compute_probability(finger_pos, string_lines):
    probs = []

    for line in string_lines:
        _, y1, _, _ = line
        dist = abs(finger_pos[1] - y1)
        prob = 1 / (dist + 1)
        probs.append(prob)

    probs = np.array(probs)
    return probs / np.sum(probs)

# =============================
# 윈도우 전체화면 설정
# =============================
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

string_lines = create_strings(w, h)

# =============================
# 메인 루프
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # 줄 기본 그리기
    for line in string_lines:
        x1, y1, x2, y2 = line
        cv2.line(display, (x1,y1), (x2,y2), (0,255,0), 2)

    # 손 인식
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    best_finger = None
    best_string = None
    best_score = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for tip_id in FINGER_TIPS:
                lm = hand_landmarks.landmark[tip_id]
                x = int(lm.x * w)
                y = int(lm.y * h)

                probs = compute_probability((x,y), string_lines)
                idx = np.argmax(probs)
                score = np.max(probs)

                if score > best_score:
                    best_score = score
                    best_finger = (x,y)
                    best_string = idx

    # 진동 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        vibrating = detect_string_vibration(prev_gray, gray, string_lines)
    else:
        vibrating = []

    for idx in vibrating:
        x1, y1, x2, y2 = string_lines[idx]
        cv2.line(display, (x1,y1),(x2,y2),(0,0,255),4)

    # 플럭 판정
    if best_finger is not None and best_string is not None:
        fx, fy = best_finger
        cv2.circle(display, (fx,fy), 10, (255,0,0), -1)

        if best_string in vibrating:
            cv2.putText(display, "PLUCK!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,0,255), 3)

    prev_gray = gray.copy()

    out.write(display)
    cv2.imshow("Result", display)

    # ESC 누르면 종료
    key = cv2.waitKey(1)
    if key == 27:
        print("ESC 눌러서 종료")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("분석 완료! output.mp4 생성됨 🎸🔥")