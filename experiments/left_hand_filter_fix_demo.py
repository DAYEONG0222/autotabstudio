import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# --- [1. 유틸리티 클래스: OneEuroFilter] ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=30):
        self.min_cutoff, self.beta, self.d_cutoff, self.freq = min_cutoff, beta, d_cutoff, freq
        self.x_prev, self.dx_prev = None, 0

    def filter(self, x):
        t_e = 1.0 / self.freq
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / t_e
        a_d = (2 * np.pi * self.d_cutoff * t_e) / (2 * np.pi * self.d_cutoff * t_e + 1)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * np.pi * cutoff * t_e) / (2 * np.pi * cutoff * t_e + 1)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

# [고도화 1] CLAHE를 이용한 국부 대비 강화 함수
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# --- [2. 초기화 및 모델 로드] ---
fret_model = YOLO(r'C:\Autotab_Studio\weights.pt')
hand_model = YOLO(r'C:\Autotab_Studio\weights_hands.pt')
mp_hands = mp.solutions.hands

# [고도화 2] LK 파라미터 최적화 (격한 움직임 대응)
lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(r'C:\Autotab_Studio\test_video2.mp4')
width, height = int(cap.get(3)), int(cap.get(4))

# 상태 변수 설정
filters = [[OneEuroFilter(beta=0.7) for _ in range(2)] for _ in range(4)]
last_valid_fingers = [] 
missing_frames_count = 0
MAX_MISSING_TOLERANCE = 20 
prev_gray, guitar_corners = None, None

STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
prev_top_line, prev_bot_line = None, None
fingertips = [8, 12, 16, 20]
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

def get_y_on_line(params, x):
    vx, vy, x0, y0 = params
    return y0 + ((x - x0) / (vx if vx != 0 else 1e-6)) * vy

# --- [3. 메인 루프] ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    display_frame = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # A. 동적 좌표 추적 (Optical Flow) 
    if guitar_corners is not None and prev_gray is not None:
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, guitar_corners, None, **lk_params)
        if status[0]: guitar_corners = p1
    prev_gray = frame_gray

    # B. Fret Zone 및 그리드 데이터 수집
    fret_results = fret_model.predict(frame, conf=0.25, verbose=False)[0]
    nut_center, all_top_pts, all_bot_pts, all_x = None, [], [], []
    fret_info = []

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
                    res = cv2.findContours(cv2.erode((m_scaled*255).astype(np.uint8), np.ones((3,3))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not res[0]: continue
                    hull = cv2.convexHull(res[0][0])
                    rect = cv2.minAreaRect(hull)
                    box = cv2.boxPoints(rect).astype(np.float32)
                    
                    # Y축 기준 정렬 로직 적용
                    dists = [np.linalg.norm(pt - nut_center) for pt in box]
                    idx = np.argsort(dists)
                    near, far = box[idx[:2]], box[idx[2:]]
                    near, far = near[np.argsort(near[:, 1])], far[np.argsort(far[:, 1])]
                    poly = np.array([near[0], far[0], far[1], near[1]], dtype="int32")
                    
                    fret_info.append({'poly': poly, 'dist': np.linalg.norm(np.mean(poly, axis=0) - nut_center)})
                    all_top_pts.extend([poly[0], poly[1]]); all_bot_pts.extend([poly[2], poly[3]]); all_x.extend([poly[0][0], poly[1][0]])

    # Huber 선형 회귀 적용
    if len(all_top_pts) >= 4:
        line_t = cv2.fitLine(np.array(all_top_pts), cv2.DIST_HUBER, 0, 0.01, 0.01)
        line_b = cv2.fitLine(np.array(all_bot_pts), cv2.DIST_HUBER, 0, 0.01, 0.01)
        curr_t, curr_b = np.array(line_t).flatten(), np.array(line_b).flatten()
        if prev_top_line is None: prev_top_line, prev_bot_line = curr_t, curr_b
        else:
            prev_top_line = (curr_t * 0.4) + (prev_top_line * 0.6)
            prev_bot_line = (curr_b * 0.4) + (prev_bot_line * 0.6)

    # --- 시각화: 그리드 및 스트링 출력 ---
    for info in fret_info:
        cv2.polylines(display_frame, [info['poly']], True, FRET_COLOR, 1, cv2.LINE_AA)

    if prev_top_line is not None:
        x_min, x_max = (min(all_x) - 50, max(all_x) + 50) if all_x else (0, width)
        for r in string_ratios:
            y1 = get_y_on_line(prev_top_line, x_min)*(1-r) + get_y_on_line(prev_bot_line, x_min)*r
            y2 = get_y_on_line(prev_top_line, x_max)*(1-r) + get_y_on_line(prev_bot_line, x_max)*r
            cv2.line(display_frame, (int(x_min), int(y1)), (int(x_max), int(y2)), STR_COLOR, 2, cv2.LINE_AA)

    # C. 손가락 추적 및 [동적 임계값] 
    hand_results = hand_model.predict(frame, conf=0.3, verbose=False)[0]
    is_tracking_success, current_fingers = False, []

    if hand_results.boxes is not None:
        for h_box in hand_results.boxes:
            if hand_model.names[int(h_box.cls[0])] != 'left_hand': continue
            x1, y1, x2, y2 = map(int, h_box.xyxy[0])
            
            # [고도화 3] ROI Crop & CLAHE 적용 
            roi = frame[max(0, y1-20):min(height, y2+20), max(0, x1-20):min(width, x2+20)]
            if roi.size == 0: continue
            enhanced_roi = apply_clahe(roi)
            
            # 상황에 따라 임계값 동적 변경
            dynamic_conf = 0.3 if missing_frames_count > 5 else 0.5 
            with mp_hands.Hands(min_detection_confidence=dynamic_conf, min_tracking_confidence=dynamic_conf) as mp_session:
                res = mp_session.process(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks:
                is_tracking_success, missing_frames_count = True, 0
                for lms in res.multi_hand_landmarks:
                    for i, tip in enumerate(fingertips):
                        raw_cx = lms.landmark[tip].x * roi.shape[1] + max(0, x1-20)
                        raw_cy = lms.landmark[tip].y * roi.shape[0] + max(0, y1-20)
                        cx, cy = filters[i][0].filter(raw_cx), filters[i][1].filter(raw_cy)
                        
                        # 속도 벡터 계산
                        vx, vy = 0, 0
                        if last_valid_fingers and len(last_valid_fingers) > i:
                            vx, vy = cx - last_valid_fingers[i]['pos'][0], cy - last_valid_fingers[i]['pos'][1]
                        current_fingers.append({'pos': (cx, cy), 'vel': (vx, vy), 'is_tracking': True})

    # D. [고도화 4] 속도 벡터 기반 관성 메모리 예측 
    if not is_tracking_success:
        missing_frames_count += 1
        if missing_frames_count <= MAX_MISSING_TOLERANCE and last_valid_fingers:
            cv2.putText(display_frame, f"PREDICTING ({missing_frames_count})", (50, 50), 2, 0.7, (0, 165, 255), 2)
            for f in last_valid_fingers:
                current_fingers.append({'pos': (f['pos'][0] + f['vel'][0]*0.8, f['pos'][1] + f['vel'][1]*0.8), 'vel': f['vel'], 'is_tracking': False})
    
    if is_tracking_success: last_valid_fingers = current_fingers.copy()

    # E. 타브 판정
    for finger in current_fingers:
        cx, cy = map(int, finger['pos'])
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
            
            if min_dist < 40:
                color = (0, 255, 255) if finger['is_tracking'] else (0, 165, 255)
                cv2.circle(display_frame, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(display_frame, f"S{best_s} F{f_num}", (cx+10, cy), 1, 1.2, color, 2)

    cv2.imshow('Bass Autotab Pro - Ultimate Integrated', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()