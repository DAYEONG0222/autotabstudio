import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 모델 및 MediaPipe 초기화
fret_model = YOLO(os.path.join(BASE_DIR, "weights.pt"))
hand_model = YOLO(os.path.join(BASE_DIR, "weights_hands.pt"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,              # 속도↑ (빠른 움직임에서 추적 안정에 도움)
    max_num_hands=1,
    min_detection_confidence=0.4,    # 재탐색 문턱↓
    min_tracking_confidence=0.4      # 트래킹 문턱↓
)

cap = cv2.VideoCapture(os.path.join(BASE_DIR, "test_video.mp4"))
width, height = int(cap.get(3)), int(cap.get(4))

# =========================
# 화면 표시 설정 (안 잘리게 + X로 닫으면 종료)
# =========================
WIN_NAME = "Bass Autotab - Length Stabilized"
TARGET_W, TARGET_H = 1280, 720

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, TARGET_W, TARGET_H)

# 시각화 설정
STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
STR_THICKNESS, FRET_THICKNESS = 2, 2
ALPHA = 0.5 

# 선과 길이를 위한 전역 변수
prev_top_line, prev_bot_line = None, None
prev_x_min, prev_x_max = None, None
fingertips = [8, 12, 16, 20]
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

def smart_sort_and_label_v6(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair, far_pair = pts[sorted_indices[:2]], pts[sorted_indices[2:]]
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]
    
    poly_pts = np.array([near_pair[0], far_pair[0], far_pair[1], near_pair[1]], dtype="float32")
    axis_vec = np.mean(pts, axis=0) - nut_center
    is_reversed = np.cross(axis_vec, poly_pts[0] - nut_center) > 0

    if not is_reversed:
        tops, bots = [poly_pts[0], poly_pts[1]], [poly_pts[3], poly_pts[2]]
    else:
        tops, bots = [poly_pts[3], poly_pts[2]], [poly_pts[0], poly_pts[1]]
    return poly_pts, tops, bots, np.max(dists)

def get_y_on_line(params, x):
    vx, vy, x0, y0 = params
    return y0 + ((x - x0) / (vx if vx != 0 else 1e-6)) * vy

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    
    fret_results = fret_model.predict(frame, conf=0.25, verbose=False)[0]
    nut_center, fret_data = None, []

    if fret_results.masks is not None:
        for i, contour in enumerate(fret_results.masks.xy):
            if fret_model.names[int(fret_results.boxes.cls[i])] == 'nut':
                nut_center = np.mean(contour, axis=0)
                break
        
        if nut_center is not None:
            for i, contour in enumerate(fret_results.masks.xy):
                if fret_model.names[int(fret_results.boxes.cls[i])] == 'fret_zone':
                    hull = cv2.convexHull(contour.astype(np.int32))
                    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
                    poly, tops, bots, d = smart_sort_and_label_v6(box, nut_center)
                    fret_data.append({'poly': poly, 'tops': tops, 'bots': bots, 'dist': d})

    fret_data.sort(key=lambda x: x['dist'])

    # --- [데이터 필터링 및 길이 안정화] ---
    all_top_pts, all_bot_pts, all_x = [], [], []
    if len(fret_data) >= 3:
        widths = [np.linalg.norm(f['tops'][0] - f['bots'][0]) for f in fret_data]
        
        for idx, f in enumerate(fret_data):
            is_valid = True
            if idx > 5:
                avg_prev_width = np.mean(widths[idx-3:idx])
                # 폭 변화가 너무 급격하면 '기울기 계산'에선 제외하되 '길이'에는 영향 안 줌
                if widths[idx] < avg_prev_width * 0.94:
                    is_valid = False
            
            if is_valid:
                all_top_pts.extend(f['tops'])
                all_bot_pts.extend(f['bots'])
            
            # 길이는 모든(필터링 전) 프렛의 x좌표를 일단 수집하여 최대치를 잡음
            all_x.extend([pt[0] for pt in f['poly']])

    if len(all_top_pts) >= 4:
        line_t = cv2.fitLine(np.array(all_top_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.001, 0.001).flatten()
        line_b = cv2.fitLine(np.array(all_bot_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.001, 0.001).flatten()
        
        if prev_top_line is None:
            prev_top_line, prev_bot_line = line_t, line_b
        else:
            prev_top_line = (line_t * ALPHA) + (prev_top_line * (1 - ALPHA))
            prev_bot_line = (line_b * ALPHA) + (prev_bot_line * (1 - ALPHA))

    # --- [길이 스무딩] ---
    if all_x:
        curr_x_min, curr_x_max = min(all_x) - 50, max(all_x) + 50
        if prev_x_min is None:
            prev_x_min, prev_x_max = curr_x_min, curr_x_max
        else:
            prev_x_min = curr_x_min * 0.1 + prev_x_min * 0.9
            prev_x_max = curr_x_max * 0.1 + prev_x_max * 0.9

    # --- [시각화: 스트링] ---
    if prev_top_line is not None and prev_x_min is not None:
        for r in string_ratios:
            p1 = (int(prev_x_min), int(get_y_on_line(prev_top_line, prev_x_min)*(1-r) + get_y_on_line(prev_bot_line, prev_x_min)*r))
            p2 = (int(prev_x_max), int(get_y_on_line(prev_top_line, prev_x_max)*(1-r) + get_y_on_line(prev_bot_line, prev_x_max)*r))
            cv2.line(display_frame, p1, p2, STR_COLOR, STR_THICKNESS, cv2.LINE_AA)

    # --- [프렛] ---
    for idx, f in enumerate(fret_data):
        cv2.polylines(display_frame, [f['poly'].astype(np.int32)], True, FRET_COLOR, FRET_THICKNESS, cv2.LINE_AA)
        cv2.putText(display_frame, f"F{idx+1}", tuple(f['poly'][0].astype(int) + [0, -10]), 0, 0.4, FRET_COLOR, 1)

    # --- [손 + MediaPipe] ---
    hand_results = hand_model.predict(frame, conf=0.3, verbose=False)[0]
    if hand_results.boxes is not None and prev_top_line is not None:
        for h_box in hand_results.boxes:
            if hand_model.names[int(h_box.cls[0])] != 'left_hand':
                continue

            x1, y1, x2, y2 = map(int, h_box.xyxy[0])

            pad = 60
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(width,  x2 + pad)
            y2p = min(height, y2 + pad)

            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0:
                continue

            mp_res = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if mp_res.multi_hand_landmarks:
                for lms in mp_res.multi_hand_landmarks:
                    for tip in fingertips:
                        cx = int(lms.landmark[tip].x * roi.shape[1] + x1p)
                        cy = int(lms.landmark[tip].y * roi.shape[0] + y1p)

                        best_s, min_dist = -1, float('inf')
                        for s_idx, r in enumerate(string_ratios):
                            target_y = get_y_on_line(prev_top_line, cx)*(1-r) + get_y_on_line(prev_bot_line, cx)*r
                            dist = abs(cy - target_y)
                            if dist < min_dist:
                                min_dist, best_s = dist, 4 - s_idx

                        current_f = 0
                        for f_num, f in enumerate(fret_data):
                            if cv2.pointPolygonTest(f['poly'], (cx, cy), False) >= 0:
                                current_f = f_num + 1
                                break

                        if min_dist < 35:
                            txt = f"S{best_s} F{current_f}"
                            (tw, th), _ = cv2.getTextSize(txt, 0, 0.5, 2)
                            cv2.rectangle(display_frame, (cx+10, cy-th-5), (cx+10+tw, cy+5), (0, 0, 0), -1)
                            cv2.putText(display_frame, txt, (cx+10, cy), 0, 0.5, (0, 255, 255), 2)
                            cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)

    # =========================
    # ✅ 화면 잘림 방지: 프레임을 창 크기에 맞춰 축소 표시
    # ✅ X(닫기) 클릭 시 즉시 종료
    # =========================
    h, w = display_frame.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h, 1.0)  # 확대는 하지 않음
    show = display_frame
    if scale < 1.0:
        show = cv2.resize(display_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow(WIN_NAME, show)

    # 창 X 버튼으로 닫혔는지 체크
    if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()