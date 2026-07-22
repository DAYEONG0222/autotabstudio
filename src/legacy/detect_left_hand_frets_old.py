import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

# [0] 환경 설정 및 GPU 확인
if torch.cuda.is_available():
    print(f"현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("⚠️ GPU를 찾지 못해 CPU로 구동합니다.")

# [1] 모델 및 디바이스 초기화
import mediapipe as mp
mp_hands = mp.solutions.hands
hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1, 
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seg_model = YOLO(r'C:\Autotab_Studio\weights.pt').to(device)

# [2] 필터 및 기하학 유틸리티
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x_prev, self.dx_prev, self.t_prev = float(x0), float(dx0), float(t0)

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = (2 * np.pi * self.d_cutoff * t_e) / (2 * np.pi * self.d_cutoff * t_e + 1)
        dx_hat = a_d * ((x - self.x_prev) / t_e) + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * np.pi * cutoff * t_e) / (2 * np.pi * cutoff * t_e + 1)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.t_prev, self.dx_prev = x_hat, t, dx_hat
        return x_hat

class PointFilter:
    def __init__(self, t0, x0, y0, min_cutoff=1.0, beta=0.005):
        self.x_filter = OneEuroFilter(t0, x0, min_cutoff=min_cutoff, beta=beta)
        self.y_filter = OneEuroFilter(t0, y0, min_cutoff=min_cutoff, beta=beta)
    def process(self, t, pt):
        return self.x_filter.filter(t, pt[0]), self.y_filter.filter(t, pt[1])

def get_y_on_line(params, x):
    if params is None or x is None: return 0
    vx, vy, x0, y0 = params
    return y0 + ((x - x0) / (vx if vx != 0 else 1e-6)) * vy

def smart_sort_and_label_v6(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair, far_pair = pts[sorted_indices[:2]], pts[sorted_indices[2:]]
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]
    poly_pts = np.array([near_pair[0], far_pair[0], far_pair[1], near_pair[1]], dtype="float32")
    axis_vec = np.mean(pts, axis=0) - nut_center
    is_reversed = np.cross(axis_vec, poly_pts[0] - nut_center) > 0
    tops, bots = ([poly_pts[3], poly_pts[2]], [poly_pts[0], poly_pts[1]]) if is_reversed else ([poly_pts[0], poly_pts[1]], [poly_pts[3], poly_pts[2]])
    return poly_pts, tops, bots, np.max(dists)

def refine_per_finger(mp_lms, roi_offset, roi_shape, nut_center, hand_mask):
    if not mp_lms or nut_center is None: return {}
    fingertips = [8, 12, 16, 20]
    
    raw_points = []
    for idx in fingertips:
        lm = mp_lms.landmark[idx]
        tx = int(lm.x * roi_shape[1] + roi_offset[0])
        ty = int(lm.y * roi_shape[0] + roi_offset[1])
        raw_points.append(np.array([tx, ty]))
    
    # 너트 거리순으로 전체 정렬 (ID Flip 방지)
    raw_points.sort(key=lambda p: np.linalg.norm(p - nut_center))
    
    result = {}
    for i, idx in enumerate(fingertips):
        pt = raw_points[i]
        in_mask = False
        if 0 <= pt[1] < hand_mask.shape[0] and 0 <= pt[0] < hand_mask.shape[1]:
            if hand_mask[pt[1], pt[0]] > 0:
                in_mask = True
        result[idx] = {'pt': pt, 'in_mask': in_mask}
    return result

# [3] 메인 루프 실행부
cap = cv2.VideoCapture(r'C:\Autotab_Studio\test_video2.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
string_ratios, fingertips = [k/8.0 for k in [0.9, 2.85, 4.85, 6.8]], [8, 12, 16, 20]

finger_filters = {}
prev_t, prev_b, px_min, px_max, last_nut = None, None, None, None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    curr_t = time.time()
    display_frame = frame.copy()
    
    # 1. YOLO Segmentation 추론
    results = seg_model.predict(frame, conf=0.3, verbose=False, device=device)[0]
    hand_mask = np.zeros((height, width), dtype=np.uint8)
    hand_detected, hand_box, fret_data = False, None, []

    if results.masks:
        for i, contour in enumerate(results.masks.xy):
            cls = seg_model.names[int(results.boxes.cls[i])]
            if cls == 'nut': 
                last_nut = np.mean(contour, axis=0)
            elif cls == 'left_hand':
                cv2.fillPoly(hand_mask, [contour.astype(np.int32)], 255)
                hand_detected, hand_box = True, results.boxes.xyxy[i].cpu().numpy()
            elif cls == 'fret_zone' and last_nut is not None:
                poly, tops, bots, d = smart_sort_and_label_v6(cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour.astype(np.int32)))), last_nut)
                fret_data.append({'poly': poly, 'tops': tops, 'bots': bots, 'dist': d})

    # 2. 지판 그리드 시각화
    fret_data.sort(key=lambda x: x['dist'])
    all_top, all_bot, all_x = [], [], []
    if len(fret_data) >= 3:
        for f in fret_data:
            all_top.extend(f['tops']); all_bot.extend(f['bots']); all_x.extend([pt[0] for pt in f['poly']])
            cv2.polylines(display_frame, [f['poly'].astype(np.int32)], True, FRET_COLOR, 2)
        if len(all_top) >= 4:
            prev_t = cv2.fitLine(np.array(all_top, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            prev_b = cv2.fitLine(np.array(all_bot, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            px_min, px_max = min(all_x)-50, max(all_x)+50

    if prev_t is not None:
        for r in string_ratios:
            p1 = (int(px_min), int(get_y_on_line(prev_t, px_min)*(1-r) + get_y_on_line(prev_b, px_min)*r))
            p2 = (int(px_max), int(get_y_on_line(prev_t, px_max)*(1-r) + get_y_on_line(prev_b, px_max)*r))
            cv2.line(display_frame, p1, p2, STR_COLOR, 1)

    # 3. 손가락 분석
    final_results = {}
    if hand_detected and last_nut is not None:
        x1, y1, x2, y2 = map(int, hand_box)
        pad = 30 # 확장된 패딩 적용
        roi_rect = (max(0, x1-pad), max(0, y1-pad))
        roi = frame[roi_rect[1]:min(height, y2+pad), roi_rect[0]:min(width, x2+pad)]
        mp_res = hands_mp.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        if mp_res.multi_hand_landmarks:
            finger_data = refine_per_finger(mp_res.multi_hand_landmarks[0], roi_rect, roi.shape, last_nut, hand_mask)
            for idx, data in finger_data.items():
                pt, in_mask = data['pt'], data['in_mask']
                if idx not in finger_filters:
                    finger_filters[idx] = PointFilter(curr_t, pt[0], pt[1])
                fx, fy = finger_filters[idx].process(curr_t, pt)
                final_results[idx] = {'coord': (int(fx), int(fy)), 'in_mask': in_mask}

    # 4. 최종 렌더링 및 타블라추어 판정
    for idx, res in final_results.items():
        cx, cy = res['coord']
        in_mask = res['in_mask']
        
        # 모든 손가락 표시를 빨간색으로 통일
        color = (0, 0, 255) 
        cv2.circle(display_frame, (cx, cy), 5, color, -1)
        
        if prev_t is not None:
            mdist, bs = float('inf'), -1
            for s_idx, r in enumerate(string_ratios):
                ty = get_y_on_line(prev_t, cx)*(1-r) + get_y_on_line(prev_b, cx)*r
                if abs(cy - ty) < mdist: mdist, bs = abs(cy - ty), 4 - s_idx
            
            cf = 0
            for f_num, f_obj in enumerate(fret_data):
                if cv2.pointPolygonTest(f_obj['poly'], (float(cx), float(cy)), False) >= 0:
                    cf = f_num + 1; break
            
            if mdist < 45:
                # 텍스트 레이블에서 정밀 모드(마스크 내부) 여부만 '*'로 표시
                label = f"S{bs} F{cf}" + ("*" if in_mask else "")
                cv2.putText(display_frame, label, (cx+10, cy), 0, 0.5, color, 2)

    cv2.imshow('Autotab V12.3 - Full Unified Color', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()