import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

# [0] 환경 설정 및 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seg_model = YOLO('weights.pt').to(device)       # 지판, 너트 분석용
hand_model = YOLO('weights_hands.pt').to(device) # 오른손 박스 검출용

# MediaPipe 초기화 (오른손 정밀 점 추출용)
import mediapipe as mp
mp_hands = mp.solutions.hands
# 모든 손가락의 정밀도를 위해 min_detection_confidence를 0.5로 설정
hands_mp = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# [1] 유틸리티 함수 (MediaPipe_Finger.py 로직 계승)
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

# [2] 비디오 소스 설정
cap = cv2.VideoCapture('test_video2.mp4')
width, height = int(cap.get(3)), int(cap.get(4))

# 파라미터 설정
string_ratios = [k/8.0 for k in [0.9, 2.85, 4.85, 6.8]]
CANDIDATE_THRESHOLD = 35 # 손가락 끝과 줄 사이의 최대 허용 거리 (픽셀)

prev_t, prev_b, last_nut = None, None, None

print("오른손 전 손가락(5개) 후보군 추출 모드 가동")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    
    # 1. 모델 추론
    res_seg = seg_model.predict(frame, conf=0.3, verbose=False, device=device)[0]
    res_hand = hand_model.predict(frame, conf=0.3, verbose=False, device=device)[0]

    fret_data, r_hand_box = [], None

    # 1-1. 지판 분석 및 스트링 연장
    if res_seg.masks:
        for i, contour in enumerate(res_seg.masks.xy):
            cls = seg_model.names[int(res_seg.boxes.cls[i])]
            if cls == 'nut': last_nut = np.mean(contour, axis=0)
            elif cls == 'fret_zone' and last_nut is not None:
                poly, tops, bots, d = smart_sort_and_label_v6(cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour.astype(np.int32)))), last_nut)
                fret_data.append({'poly': poly, 'tops': tops, 'bots': bots, 'dist': d})

    # 1-2. 오른손 탐지 (화면 1/4 공간 필터 적용)
    if len(res_hand.boxes) > 0:
        for box in res_hand.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            if ((bx1 + bx2) / 2) < (width / 4): 
                r_hand_box = [bx1, by1, bx2, by2]; break

    # 2. 스트링 라인 피팅 (전체 영역 연장)
    fret_data.sort(key=lambda x: x['dist'])
    all_top, all_bot = [], []
    if len(fret_data) >= 3:
        for f in fret_data: all_top.extend(f['tops']); all_bot.extend(f['bots'])
        if len(all_top) >= 4:
            prev_t = cv2.fitLine(np.array(all_top, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            prev_b = cv2.fitLine(np.array(all_bot, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    # 3. [핵심] 다섯 손가락 MediaPipe 점 추출 및 후보군 선정
    candidate_strings = set()
    if r_hand_box is not None and prev_t is not None:
        rx1, ry1, rx2, ry2 = map(int, r_hand_box)
        roi_pad = 20
        roi_rect = (max(0, rx1-roi_pad), max(0, ry1-roi_pad))
        roi_img = frame[roi_rect[1]:min(height, ry2+roi_pad), roi_rect[0]:min(width, rx2+roi_pad)]
        
        mp_res = hands_mp.process(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        
        if mp_res.multi_hand_landmarks:
            for hand_lms in mp_res.multi_hand_landmarks:
                # 다섯 손가락 끝점 인덱스: 엄지(4), 검지(8), 중지(12), 약지(16), 소지(20)
                fingertip_indices = [4, 8, 12, 16, 20]
                
                for idx in fingertip_indices:
                    lm = hand_lms.landmark[idx]
                    cx = int(lm.x * roi_img.shape[1] + roi_rect[0])
                    cy = int(lm.y * roi_img.shape[0] + roi_rect[1])
                    
                    # 모든 손가락 끝점 시각화 (보라색 점)
                    cv2.circle(display_frame, (cx, cy), 5, (255, 0, 255), -1)

                    # 연장된 스트링 라인과의 거리 계산
                    for s_idx, r in enumerate(string_ratios):
                        line_y = get_y_on_line(prev_t, cx)*(1-r) + get_y_on_line(prev_b, cx)*r
                        dist = abs(cy - line_y)
                        
                        # 거리가 임계값 이내라면 후보군으로 추가
                        if dist < CANDIDATE_THRESHOLD:
                            candidate_strings.add(4 - s_idx)
                            # 시각화: 후보가 된 줄을 강조 (하늘색 선)
                            cv2.line(display_frame, (cx-20, int(line_y)), (cx+20, int(line_y)), (255, 255, 0), 2)

        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(display_frame, f"Candidates: {sorted(list(candidate_strings))}", (rx1, ry1-15), 0, 0.6, (255, 255, 0), 2)

    # 4. 필터 라인 및 가이드 렌더링
    cv2.line(display_frame, (int(width/4), 0), (int(width/4), height), (255, 255, 255), 1)
    if prev_t is not None:
        for r in string_ratios:
            p1 = (0, int(get_y_on_line(prev_t, 0)*(1-r) + get_y_on_line(prev_b, 0)*r))
            p2 = (width, int(get_y_on_line(prev_t, width)*(1-r) + get_y_on_line(prev_b, width)*r))
            # 쨍한 녹색 가이드라인
            cv2.line(display_frame, p1, p2, (0, 255, 0), 2)

    cv2.imshow('Autotab V26 - Full Hand Spatial Filter', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()