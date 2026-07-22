import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# 1. 모델 및 MediaPipe 초기화
fret_model = YOLO(r'C:\Autotab_Studio\weights.pt')
hand_model = YOLO(r'C:\Autotab_Studio\weights_hands.pt')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(r'C:\Autotab_Studio\test_video9.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🛠️ 시각화 및 알고리즘 설정
STR_COLOR = (50, 255, 50)      
FRET_COLOR = (255, 255, 100)   
STR_THICKNESS = 2
FRET_THICKNESS = 2
ALPHA = 0.4 # 오므라듬 방지를 위해 조금 더 강한 스무딩

prev_top_line, prev_bot_line = None, None
fingertips = [8, 12, 16, 20]

custom_k_values = [0.9, 2.85, 4.85, 6.8]
string_ratios = [k / 8.0 for k in custom_k_values]

def smart_sort_and_label_v4(pts, nut_center):
    """
    1. B코드의 Y정렬로 나비넥타이 방지
    2. 벡터 투영으로 진짜 상/하단 판별 (줄뜸 방지)
    """
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair = pts[sorted_indices[:2]]
    far_pair = pts[sorted_indices[2:]]
    
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]
    
    # 나비넥타이 방지용 시각적 폴리곤
    poly_pts = np.array([near_pair[0], far_pair[0], far_pair[1], near_pair[1]], dtype="float32")
    
    # 줄뜸 방지용 상/하 분리 (축 벡터 투영)
    fret_center = np.mean(pts, axis=0)
    axis_vec = fret_center - nut_center
    is_reversed = np.cross(axis_vec, poly_pts[0] - nut_center) > 0

    if not is_reversed:
        tops = [poly_pts[0], poly_pts[1]]
        bots = [poly_pts[3], poly_pts[2]]
    else:
        tops = [poly_pts[3], poly_pts[2]]
        bots = [poly_pts[0], poly_pts[1]]
        
    return poly_pts, tops, bots

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    display_frame = frame.copy()
    
    fret_results = fret_model.predict(frame, conf=0.25, verbose=False)[0]
    nut_center, fret_info, all_top_pts, all_bot_pts, all_x = None, [], [], [], []

    if fret_results.masks is not None:
        for i, contour in enumerate(fret_results.masks.xy):
            if fret_model.names[int(fret_results.boxes.cls[i])] == 'nut':
                nut_center = np.mean(contour, axis=0)
                break
        
        if nut_center is not None:
            for i, contour in enumerate(fret_results.masks.xy):
                if fret_model.names[int(fret_results.boxes.cls[i])] == 'fret_zone':
                    # [오므라듬 방지 핵심 1] Erosion은 빼고 Convex Hull만 적용
                    cnt = contour.astype(np.int32)
                    hull = cv2.convexHull(cnt)
                    
                    rect = cv2.minAreaRect(hull)
                    box = cv2.boxPoints(rect).astype(np.float32)
                    
                    poly, tops, bots = smart_sort_and_label_v4(box, nut_center)
                    
                    fret_info.append({
                        'poly': poly.astype(np.int32), 
                        'dist': np.linalg.norm(np.mean(poly, axis=0) - nut_center)
                    })
                    all_top_pts.extend(tops)
                    all_bot_pts.extend(bots)
                    all_x.extend([pt[0] for pt in poly])

    fret_info.sort(key=lambda x: x['dist'])

    if len(all_top_pts) >= 4:
        # [오므라듬 방지 핵심 2] Huber Loss 적용
        line_t = cv2.fitLine(np.array(all_top_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        line_b = cv2.fitLine(np.array(all_bot_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        
        curr_t, curr_b = np.array(line_t).flatten(), np.array(line_b).flatten()
        if prev_top_line is None: prev_top_line, prev_bot_line = curr_t, curr_b
        else:
            prev_top_line = (curr_t * ALPHA) + (prev_top_line * (1 - ALPHA))
            prev_bot_line = (curr_b * ALPHA) + (prev_bot_line * (1 - ALPHA))

    # --- [시각화: 프렛 및 스트링] ---
    for idx, info in enumerate(fret_info):
        cv2.polylines(display_frame, [info['poly']], True, FRET_COLOR, FRET_THICKNESS, cv2.LINE_AA)
        cv2.putText(display_frame, f"F{idx+1}", tuple(info['poly'][0] + [0, -10]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FRET_COLOR, 1, cv2.LINE_AA)

    def get_y_on_line(params, x):
        vx, vy, x0, y0 = params
        return y0 + ((x - x0) / vx) * vy

    if prev_top_line is not None:
        x_min, x_max = (min(all_x) - 50, max(all_x) + 50) if all_x else (0, width)
        for r in string_ratios:
            p1 = (int(x_min), int(get_y_on_line(prev_top_line, x_min)*(1-r) + get_y_on_line(prev_bot_line, x_min)*r))
            p2 = (int(x_max), int(get_y_on_line(prev_top_line, x_max)*(1-r) + get_y_on_line(prev_bot_line, x_max)*r))
            cv2.line(display_frame, p1, p2, STR_COLOR, STR_THICKNESS, cv2.LINE_AA)

    # --- [손가락 추적 및 타브 판정 (기존 기능 유지)] ---
    hand_results = hand_model.predict(frame, conf=0.3, verbose=False)[0]
    if hand_results.boxes is not None and prev_top_line is not None:
        for h_box in hand_results.boxes:
            if hand_model.names[int(h_box.cls[0])] != 'left_hand': continue
            x1, y1, x2, y2 = map(int, h_box.xyxy[0])
            roi = frame[max(0, y1-20):min(height, y2+20), max(0, x1-20):min(width, x2+20)]
            if roi.size == 0: continue
            mp_res = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if mp_res.multi_hand_landmarks:
                for lms in mp_res.multi_hand_landmarks:
                    for tip in fingertips:
                        cx = int(lms.landmark[tip].x * roi.shape[1] + max(0, x1-20))
                        cy = int(lms.landmark[tip].y * roi.shape[0] + max(0, y1-20))
                        best_s, min_dist = -1, float('inf')
                        for s_idx, r in enumerate(string_ratios):
                            target_y = get_y_on_line(prev_top_line, cx)*(1-r) + get_y_on_line(prev_bot_line, cx)*r
                            dist = abs(cy - target_y)
                            if dist < min_dist: min_dist, best_s = dist, 4 - s_idx
                        current_f = 0
                        for f_num, info in enumerate(fret_info):
                            if cv2.pointPolygonTest(info['poly'], (cx, cy), False) >= 0:
                                current_f = f_num + 1; break
                        if min_dist < 35:
                            txt = f"S{best_s} F{current_f}"
                            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(display_frame, (cx+10, cy-th-5), (cx+10+tw, cy+5), (0,0,0), -1)
                            cv2.putText(display_frame, txt, (cx+10, cy), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)

    cv2.imshow('Bass Autotab - No Pinch Final', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()