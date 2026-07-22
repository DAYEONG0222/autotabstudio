from ultralytics import YOLO
import cv2
import numpy as np
import time

# -----------------------------------------------------------
# 🛠️ 설정
# -----------------------------------------------------------
model = YOLO(r'C:\Autotab_Studio\weights.pt')
video_path = r'C:\Autotab_Studio\test_video.mp4'
cap = cv2.VideoCapture(video_path)

lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

is_tracking = False
p0 = None
old_gray = None
current_boundary_funcs = [] 
ZONE_BOUNDARIES = [0.0, 0.25, 0.50, 0.75, 1.0]

# -----------------------------------------------------------
# 🎛️ [하이브리드 감도 설정]
# -----------------------------------------------------------
noise_floor = [1.0, 1.0, 1.0, 1.0] 

# [수정 1] 민감도 더 낮춤 (평소보다 1.5배만 커도 감지)
ADAPTIVE_SENSITIVITY = 1.5
MIN_THRESHOLD = 20.0

prev_string_scores = [0, 0, 0, 0]
last_pick_time = [0, 0, 0, 0]
PICK_COOLDOWN = 0.1 # 쿨타임 단축
prev_time = 0

# [수정 2] 메시지 유지 시간 2배 늘림 (1.0초)
display_message = ""          
display_start_time = 0        
DISPLAY_DURATION = 1.0 

# -----------------------------------------------------------
# 🛠️ 유틸리티 함수
# -----------------------------------------------------------
def smart_sort_corners(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair = [pts[sorted_indices[0]], pts[sorted_indices[1]]]
    far_pair = [pts[sorted_indices[2]], pts[sorted_indices[3]]]
    fret_center = np.mean(pts, axis=0)
    axis_vec = fret_center - nut_center
    def is_above_axis(pt, origin, axis):
        pt_vec = pt - origin
        cross_val = axis[0] * pt_vec[1] - axis[1] * pt_vec[0]
        return cross_val < 0 
    if is_above_axis(near_pair[0], nut_center, axis_vec): near_top, near_bot = near_pair[0], near_pair[1]
    else: near_top, near_bot = near_pair[1], near_pair[0]
    if is_above_axis(far_pair[0], nut_center, axis_vec): far_top, far_bot = far_pair[0], far_pair[1]
    else: far_top, far_bot = far_pair[1], far_pair[0]
    return np.array([near_top, far_top, far_bot, near_bot], dtype="float32")

def get_y_from_line(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) < 1e-3: return y1 
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

print("▶ 하이브리드 모드 시작: 순간 타격 + 지속 진동 모두 감지")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    curr_time = time.time()
    time_diff = curr_time - prev_time
    if prev_time != 0 and time_diff > 0: fps = 1 / time_diff
    else: fps = 0
    prev_time = curr_time
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame_gray.shape
    current_boundary_funcs = [] 

    if is_tracking and p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None and len(p1) == 4:
            p_st, p_sb, p_et, p_eb = p1.reshape(4, 2)
            for pt in p1: cv2.circle(frame, tuple(pt.ravel().astype(int)), 5, (0, 255, 0), -1)
            for ratio in ZONE_BOUNDARIES:
                current_boundary_funcs.append(lambda x, r=ratio, t1=p_st, t2=p_et, b1=p_sb, b2=p_eb: get_y_from_line(t1, t2, x) * (1 - r) + get_y_from_line(b1, b2, x) * r)
            for func in current_boundary_funcs: cv2.line(frame, (0, int(func(0))), (w, int(func(w))), (255, 255, 0), 2)

            if old_gray is not None:
                frame_diff = cv2.absdiff(old_gray, frame_gray)
                _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                
                current_scores = []
                roi_x_start = int(p_st[0] * 0.4 + p_et[0] * 0.6)
                roi_x_end = int(p_et[0] + 50)
                roi_x_start = max(0, roi_x_start)
                roi_x_end = min(w, roi_x_end)

                for i in range(4):
                    mask = np.zeros_like(frame_gray)
                    pts = np.array([
                        [roi_x_start, int(current_boundary_funcs[i](roi_x_start))],
                        [roi_x_end,   int(current_boundary_funcs[i](roi_x_end))],
                        [roi_x_end,   int(current_boundary_funcs[i+1](roi_x_end))],
                        [roi_x_start, int(current_boundary_funcs[i+1](roi_x_start))]
                    ], np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    masked_motion = cv2.bitwise_and(thresh, thresh, mask=mask)
                    score = cv2.countNonZero(masked_motion)
                    current_scores.append(score)

                impulses = []
                for i in range(4):
                    impulse = current_scores[i] - prev_string_scores[i]
                    impulses.append(max(0, impulse))

                # ----------------------------------------------------------
                # 🧪 [하이브리드 판정 로직]
                # ----------------------------------------------------------
                candidates = [] 
                
                for i in range(4):
                    dynamic_threshold = max(noise_floor[i] * ADAPTIVE_SENSITIVITY, MIN_THRESHOLD)
                    
                    # 조건 1: 급격한 변화(Impulse)가 감지됨
                    is_impulse = impulses[i] > dynamic_threshold
                    
                    # 조건 2: 그냥 진동(Score) 자체가 큼 (Sustain)
                    # -> 노이즈 기준값의 2배 이상이면 지속음으로 인정
                    is_sustaining = current_scores[i] > (dynamic_threshold * 2.0)
                    
                    # 둘 중 하나라도 만족하면 후보 등록
                    if is_impulse or is_sustaining:
                        # 점수는 둘 중 더 큰 값으로 사용
                        final_score = max(impulses[i], current_scores[i])
                        candidates.append((i, final_score))

                detected_string = -1

                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    best_idx, best_score = candidates[0]
                    
                    # 쿨타임 체크 (너무 자주 바뀌는 것 방지)
                    if curr_time - last_pick_time[best_idx] > PICK_COOLDOWN:
                        detected_string = best_idx + 1
                        last_pick_time[best_idx] = curr_time
                        
                        display_message = f"PICK!: String {detected_string}"
                        display_start_time = curr_time
                        
                        print(f"✅ DETECTED: {display_message} (Score:{best_score})")

                        idx = detected_string - 1
                        overlay = frame.copy()
                        full_pts = np.array([
                            [0, int(current_boundary_funcs[idx](0))],
                            [w, int(current_boundary_funcs[idx](w))],
                            [w, int(current_boundary_funcs[idx+1](w))],
                            [0, int(current_boundary_funcs[idx+1](0))]
                        ], np.int32)
                        cv2.fillPoly(overlay, [full_pts], (0, 0, 255))
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # 노이즈 학습
                for i in range(4):
                    if i + 1 != detected_string:
                        noise_floor[i] = (noise_floor[i] * 0.95) + (current_scores[i] * 0.05)

                # ----------------------------------------------------------
                # 📺 [화면 표시]
                # ----------------------------------------------------------
                # 메시지 유지 시간이 지나지 않았으면 계속 표시
                if display_message and (curr_time - display_start_time < DISPLAY_DURATION):
                    cv2.rectangle(frame, (40, 100), (500, 180), (0, 0, 0), -1)
                    cv2.putText(frame, display_message, (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                # 디버깅: 점수(S)와 기준선(T) 표시
                for i in range(4):
                    s = current_scores[i]
                    thresh_val = int(max(noise_floor[i] * ADAPTIVE_SENSITIVITY, MIN_THRESHOLD))
                    # 활성화 상태면 빨간색
                    color = (0, 0, 255) if s > (thresh_val * 2.0) or impulses[i] > thresh_val else (0, 255, 0)
                    cv2.putText(frame, f"S{i+1}: {s}/{thresh_val}", (10, h - 150 + i*35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                prev_string_scores = current_scores

            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)
        else:
            print("❌ 추적 실패.")
            is_tracking = False
            p0 = None
    else:
        # 스캔 모드
        cv2.putText(frame, "SCANNING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
        result = results[0]
        all_top_points = [] 
        all_bot_points = [] 
        all_x_coords = []
        nut_center = None

        if result.masks is not None:
            contours = result.masks.xy
            for i, contour in enumerate(contours):
                if len(contour) > 0 and model.names[int(result.boxes.cls[i])] == 'nut':
                    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.int32)))
                    nut_center = np.mean(box, axis=0)
            if nut_center is not None:
                for i, contour in enumerate(contours):
                    if len(contour) > 0 and model.names[int(result.boxes.cls[i])] == 'fret_zone':
                        cnt = contour.astype(np.int32)
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect).astype(np.float32)
                        sorted_pts = smart_sort_corners(box, nut_center)
                        all_top_points.extend([sorted_pts[0], sorted_pts[1]])
                        all_bot_points.extend([sorted_pts[2], sorted_pts[3]])
                        all_x_coords.extend([pt[0] for pt in sorted_pts])

        if len(all_top_points) >= 6:
            pts_top = np.array(all_top_points, dtype=np.int32)
            line_top = cv2.fitLine(pts_top, cv2.DIST_L2, 0, 0.01, 0.01)
            vx_t, vy_t, x0_t, y0_t = [val.item() for val in line_top]
            pts_bot = np.array(all_bot_points, dtype=np.int32)
            line_bot = cv2.fitLine(pts_bot, cv2.DIST_L2, 0, 0.01, 0.01)
            vx_b, vy_b, x0_b, y0_b = [val.item() for val in line_bot]
            
            def get_point(vx, vy, x0, y0, x):
                if abs(vx) < 1e-3: return np.array([x0, y0], dtype=np.float32)
                t = (x - x0) / vx
                y = y0 + t * vy
                return np.array([x, y], dtype=np.float32)

            x_min, x_max = min(all_x_coords), max(all_x_coords)
            p_st = get_point(vx_t, vy_t, x0_t, y0_t, x_min)
            p_sb = get_point(vx_b, vy_b, x0_b, y0_b, x_min)
            p_et = get_point(vx_t, vy_t, x0_t, y0_t, x_max)
            p_eb = get_point(vx_b, vy_b, x0_b, y0_b, x_max)
            
            p0 = np.array([p_st, p_sb, p_et, p_eb], dtype=np.float32).reshape(-1, 1, 2)
            old_gray = frame_gray.copy()
            is_tracking = True
            
            prev_string_scores = [0, 0, 0, 0]
            last_pick_time = [0, 0, 0, 0]
            noise_floor = [1.0, 1.0, 1.0, 1.0]
            prev_time = time.time()
            print(">>> 🚀 Auto-Lock! 트래킹 시작")

    cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Autotab Hybrid Mode', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()