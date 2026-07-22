import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
import mediapipe as mp

# [0] 환경 설정 및 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seg_model = YOLO('weights.pt').to(device)
hand_model = YOLO('weights_hands.pt').to(device)

mp_hands = mp.solutions.hands
hands_mp = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# [1] 지터링 방지를 위한 원유로 필터 (One Euro Filter) 클래스
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        
        # 속도 계산 및 필터링
        a_d = (2 * np.pi * self.d_cutoff * t_e) / (2 * np.pi * self.d_cutoff * t_e + 1)
        dx_hat = a_d * ((x - self.x_prev) / t_e) + (1 - a_d) * self.dx_prev
        
        # 컷오프 주파수 결정 (지터링 억제의 핵심)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * np.pi * cutoff * t_e) / (2 * np.pi * cutoff * t_e + 1)
        
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.t_prev, self.dx_prev = x_hat, t, dx_hat
        return x_hat

# [2] 유틸리티 함수
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

# [3] 필터 및 데이터 저장소 초기화
finger_filters = {}  # 손가락별 원유로 필터 객체 저장
finger_history = {4: [], 8: [], 12: [], 16: []}
HISTORY_LEN = 6 
MOVE_THRESHOLD = 3.5 # 필터 적용 후이므로 임계값을 조금 더 엄격하게 조정 가능

cap = cv2.VideoCapture('test_video3.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
string_ratios = [k/8.0 for k in [0.9, 2.85, 4.85, 6.8]]
CANDIDATE_THRESHOLD = 35 

prev_t, prev_b, last_nut = None, None, None

print("지터링 방지 필터가 적용된 통합 시스템 가동 중...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    curr_time = time.time()
    display_frame = frame.copy()
    
    # 1. 모델 추론
    res_seg = seg_model.predict(frame, conf=0.3, verbose=False, device=device)[0]
    res_hand = hand_model.predict(frame, conf=0.3, verbose=False, device=device)[0]

    fret_data, r_hand_box = [], None

    # 지판 분석 및 스트링 연장 로직
    if res_seg.masks:
        for i, contour in enumerate(res_seg.masks.xy):
            cls = seg_model.names[int(res_seg.boxes.cls[i])]
            if cls == 'nut': last_nut = np.mean(contour, axis=0)
            elif cls == 'fret_zone' and last_nut is not None:
                poly, tops, bots, d = smart_sort_and_label_v6(cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour.astype(np.int32)))), last_nut)
                fret_data.append({'poly': poly, 'tops': tops, 'bots': bots, 'dist': d})

    # 오른손 탐지 (화면 1/4 구역)
    if len(res_hand.boxes) > 0:
        for box in res_hand.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            if ((bx1 + bx2) / 2) < (width / 4): 
                r_hand_box = [bx1, by1, bx2, by2]; break

    if len(fret_data) >= 3:
        fret_data.sort(key=lambda x: x['dist'])
        all_top, all_bot = [], []
        for f in fret_data: all_top.extend(f['tops']); all_bot.extend(f['bots'])
        if len(all_top) >= 4:
            prev_t = cv2.fitLine(np.array(all_top, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            prev_b = cv2.fitLine(np.array(all_bot, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    # 2. [핵심] 필터링된 미디어파이프 점 및 후보군 분석
    candidate_strings = set()
    if r_hand_box is not None and prev_t is not None:
        rx1, ry1, rx2, ry2 = map(int, r_hand_box)
        roi_rect = (max(0, rx1-20), max(0, ry1-20))
        roi_img = frame[roi_rect[1]:min(height, ry2+20), roi_rect[0]:min(width, rx2+20)]
        mp_res = hands_mp.process(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        
        if mp_res.multi_hand_landmarks:
            for hand_lms in mp_res.multi_hand_landmarks:
                for idx in [4, 8, 12, 16]:
                    lm = hand_lms.landmark[idx]
                    raw_cy = int(lm.y * roi_img.shape[0] + roi_rect[1])
                    cx = int(lm.x * roi_img.shape[1] + roi_rect[0])
                    
                    # 지터링 방지 필터 적용
                    if idx not in finger_filters:
                        finger_filters[idx] = OneEuroFilter(curr_time, raw_cy, min_cutoff=1.0, beta=0.01)
                    
                    # 평활화된 Y좌표 추출
                    filtered_cy = finger_filters[idx].filter(curr_time, raw_cy)
                    
                    # 히스토리 업데이트
                    finger_history[idx].append(filtered_cy)
                    if len(finger_history[idx]) > HISTORY_LEN: finger_history[idx].pop(0)
                    
                    # 움직임 계산 (필터링된 값의 표준편차)
                    movement = 0
                    if len(finger_history[idx]) >= 3:
                        movement = np.std(finger_history[idx])
                    
                    is_active = movement > MOVE_THRESHOLD
                    point_color = (255, 0, 255) if is_active else (150, 150, 150)
                    cv2.circle(display_frame, (cx, int(filtered_cy)), 6, point_color, -1)

                    for s_idx, r in enumerate(string_ratios):
                        line_y = get_y_on_line(prev_t, cx)*(1-r) + get_y_on_line(prev_b, cx)*r
                        if abs(filtered_cy - line_y) < CANDIDATE_THRESHOLD:
                            if is_active:
                                candidate_strings.add(4 - s_idx)
                                cv2.line(display_frame, (cx-20, int(line_y)), (cx+20, int(line_y)), (0, 255, 255), 2)
                            else:
                                cv2.putText(display_frame, "Static", (cx, int(filtered_cy)-12), 0, 0.4, (150, 150, 150), 1)

        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(display_frame, f"Active Cand: {sorted(list(candidate_strings))}", (rx1, ry1-15), 0, 0.6, (0, 255, 255), 2)

    # 3. 가이드 렌더링
    cv2.line(display_frame, (int(width/4), 0), (int(width/4), height), (255, 255, 255), 1)
    if prev_t is not None:
        for r in string_ratios:
            p1 = (0, int(get_y_on_line(prev_t, 0)*(1-r) + get_y_on_line(prev_b, 0)*r))
            p2 = (width, int(get_y_on_line(prev_t, width)*(1-r) + get_y_on_line(prev_b, width)*r))
            cv2.line(display_frame, p1, p2, (0, 255, 0), 2)

    cv2.imshow('Autotab V29 - Anti-Jittering System', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()