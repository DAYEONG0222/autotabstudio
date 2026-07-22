import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# --- [1. 유틸리티 클래스: OneEuroFilter] ---
class OneEuroFilter:
    """좌표 떨림을 방지하고 속도에 따라 유연하게 반응하는 필터 [cite: 3]"""
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=30):
        self.min_cutoff, self.beta, self.d_cutoff, self.freq = min_cutoff, beta, d_cutoff, freq
        self.x_prev, self.dx_prev = None, 0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x):
        t_e = 1.0 / self.freq
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / t_e
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

# --- [2. 함수 정의: 감마 조절 및 정렬] ---
def adjust_gamma(image, gamma=1.5):
    """이미지 밝기를 강제로 부스팅하여 인식률 향상 """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def smart_sort_corners(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair, far_pair = pts[sorted_indices[:2]], pts[sorted_indices[2:]]
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]
    return np.array([near_pair[0], far_pair[0], far_pair[1], near_pair[1]], dtype="float32")

def get_y_on_line(params, x):
    vx, vy, x0, y0 = params
    return y0 + ((x - x0) / (vx if vx != 0 else 1e-6)) * vy

# --- [3. 초기화] ---
fret_model = YOLO(r'C:\Autotab_Studio\weights.pt')
hand_model = YOLO(r'C:\Autotab_Studio\weights_hands.pt')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.3, min_tracking_confidence=0.3 # 저임계값 설정 
)

cap = cv2.VideoCapture(r'C:\Autotab_Studio\test_video2.mp4')
width, height = int(cap.get(3)), int(cap.get(4))

# 보정용 상태 변수
filters = [[OneEuroFilter(beta=0.7) for _ in range(2)] for _ in range(4)]
MAX_MISSING_TOLERANCE = 30 # [cite: 4]
missing_frames_count = 0
last_valid_fingers = []

# LK 광학 흐름 설정 [cite: 6]
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_gray, guitar_corners = None, None

STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
STR_THICKNESS, FRET_THICKNESS = 2, 2
ALPHA = 0.4
prev_top_line, prev_bot_line = None, None
fingertips = [8, 12, 16, 20]
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

# --- [4. 메인 루프] ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    display_frame = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # A. 동적 좌표 추적 (Optical Flow) [cite: 5]
    if guitar_corners is not None and prev_gray is not None:
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, guitar_corners, None, **lk_params)
        if status[0]: guitar_corners = p1
    prev_gray = frame_gray

    # B. Fret/Nut 검출 및 그리드 생성
    fret_results = fret_model.predict(frame, conf=0.25, verbose=False)[0]
    nut_center, fret_info, all_top_pts, all_bot_pts, all_x = None, [], [], [], []

    if fret_results.masks is not None:
        masks_data = fret_results.masks.data.cpu().numpy()
        for i, mask in enumerate(masks_data):
            if fret_model.names[int(fret_results.boxes.cls[i])] == 'nut':
                m_scaled = cv2.resize(mask, (width, height))
                contours, _ = cv2.findContours((m_scaled*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    nut_center = np.mean(contours[0], axis=0)[0]
                    if guitar_corners is None: guitar_corners = np.array([nut_center], dtype=np.float32).reshape(-1, 1, 2)
                break
        
        if nut_center is not None:
            for i, mask in enumerate(masks_data):
                if fret_model.names[int(fret_results.boxes.cls[i])] == 'fret_zone':
                    m_scaled = cv2.resize(mask, (width, height))
                    eroded = cv2.erode((m_scaled*255).astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
                    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours: continue
                    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
                    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
                    sorted_pts = smart_sort_corners(box, nut_center)
                    fret_info.append({'poly': sorted_pts.astype(np.int32), 'dist': np.linalg.norm(np.mean(sorted_pts, axis=0) - nut_center)})
                    all_top_pts.extend([sorted_pts[0], sorted_pts[1]]); all_bot_pts.extend([sorted_pts[2], sorted_pts[3]]); all_x.extend([pt[0] for pt in sorted_pts])

    # C. 스트링 라인 계산 (Huber 적용)
    if len(all_top_pts) >= 4:
        line_t = cv2.fitLine(np.array(all_top_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        line_b = cv2.fitLine(np.array(all_bot_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        curr_t, curr_b = np.array(line_t).flatten(), np.array(line_b).flatten()
        if prev_top_line is None: prev_top_line, prev_bot_line = curr_t, curr_b
        else:
            prev_top_line = (curr_t * ALPHA) + (prev_top_line * (1 - ALPHA))
            prev_bot_line = (curr_b * ALPHA) + (prev_bot_line * (1 - ALPHA))

    # D. 시각화 (그리드 및 스트링)
    for idx, info in enumerate(sorted(fret_info, key=lambda x: x['dist'])):
        cv2.polylines(display_frame, [info['poly']], True, FRET_COLOR, 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"F{idx+1}", tuple(info['poly'][0] + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, FRET_COLOR, 1)

    if prev_top_line is not None:
        x_min, x_max = (min(all_x) - 50, max(all_x) + 50) if all_x else (0, width)
        for r in string_ratios:
            y1 = get_y_on_line(prev_top_line, x_min)*(1-r) + get_y_on_line(prev_bot_line, x_min)*r
            y2 = get_y_on_line(prev_top_line, x_max)*(1-r) + get_y_on_line(prev_bot_line, x_max)*r
            cv2.line(display_frame, (int(x_min), int(y1)), (int(x_max), int(y2)), STR_COLOR, STR_THICKNESS, cv2.LINE_AA)

    # E. 손가락 추적 (조건부 재귀 추론 및 관성 메모리)
    hand_results = hand_model.predict(frame, conf=0.3, verbose=False)[0]
    current_fingers, is_tracking_success = [], False

    if hand_results.boxes is not None:
        for h_box in hand_results.boxes:
            if hand_model.names[int(h_box.cls[0])] != 'left_hand': continue
            x1, y1, x2, y2 = map(int, h_box.xyxy[0])
            roi = frame[max(0, y1-20):min(height, y2+20), max(0, x1-20):min(width, x2+20)]
            if roi.size == 0: continue
            
            # 1차 인식 -> 실패 시 감마 부스팅 2차 인식 
            mp_res = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if not mp_res.multi_hand_landmarks:
                mp_res = hands.process(cv2.cvtColor(adjust_gamma(roi, 1.5), cv2.COLOR_BGR2RGB))

            if mp_res.multi_hand_landmarks:
                is_tracking_success = True
                missing_frames_count = 0
                for lms in mp_res.multi_hand_landmarks:
                    for i, tip in enumerate(fingertips):
                        raw_cx = lms.landmark[tip].x * roi.shape[1] + max(0, x1-20)
                        raw_cy = lms.landmark[tip].y * roi.shape[0] + max(0, y1-20)
                        cx, cy = int(filters[i][0].filter(raw_cx)), int(filters[i][1].filter(raw_cy)) # 필터 적용 
                        current_fingers.append({'pos': (cx, cy), 'is_tracking': True})

    # F. 관성 메모리 적용 (Lost 시 추정값 반환) 
    if not is_tracking_success:
        missing_frames_count += 1
        if missing_frames_count <= MAX_MISSING_TOLERANCE and last_valid_fingers:
            cv2.putText(display_frame, "MEMORY (Lost)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            current_fingers = last_valid_fingers
    else: last_valid_fingers = current_fingers.copy()

    # G. 판정 시각화
    for finger in current_fingers:
        cx, cy = finger['pos']
        if prev_top_line is not None:
            best_s, min_dist = -1, float('inf')
            for s_idx, r in enumerate(string_ratios):
                target_y = get_y_on_line(prev_top_line, cx)*(1-r) + get_y_on_line(prev_bot_line, cx)*r
                dist = abs(cy - target_y)
                if dist < min_dist: min_dist, best_s = dist, 4 - s_idx
            
            f_num = 0
            for i, info in enumerate(sorted(fret_info, key=lambda x: x['dist'])):
                if cv2.pointPolygonTest(info['poly'], (cx, cy), False) >= 0:
                    f_num = i + 1; break
            
            if min_dist < 35:
                cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(display_frame, f"S{best_s} F{f_num}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow('Bass Autotab Pro - Ultimate Robust', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()