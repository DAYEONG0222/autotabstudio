from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r'C:\Autotab_Studio\weights.pt')
video_path = r'C:\Autotab_Studio\test_video2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_dual_regression.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# -----------------------------------------------------------
# 🛠️ [SMART SORT] 너트 기준 벡터 정렬
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
        return np.cross(axis, pt_vec) < 0 

    if is_above_axis(near_pair[0], nut_center, axis_vec):
        near_top, near_bot = near_pair[0], near_pair[1]
    else:
        near_top, near_bot = near_pair[1], near_pair[0]
        
    if is_above_axis(far_pair[0], nut_center, axis_vec):
        far_top, far_bot = far_pair[0], far_pair[1]
    else:
        far_top, far_bot = far_pair[1], far_pair[0]

    return np.array([near_top, far_top, far_bot, near_bot], dtype="float32")

# -----------------------------------------------------------
# 🛠️ 안정화 변수 (두 개의 라인을 각각 기억)
# -----------------------------------------------------------
# Top Line [vx, vy, x0, y0]
prev_top_line = None
# Bot Line [vx, vy, x0, y0]
prev_bot_line = None
alpha = 0.5

print("▶ Dual Regression 적용: 위/아래 라인을 따로 계산하여 원근감을 반영합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    overlay = frame.copy()
    results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
    result = results[0]

    all_top_points = [] # 1번줄 쪽 점들 모음
    all_bot_points = [] # 4번줄 쪽 점들 모음
    all_x_coords = []   # 그리기 범위 결정을 위한 X좌표 모음

    nut_center = None

    if result.masks is not None:
        contours = result.masks.xy
        
        # 1. Nut 찾기
        for i, contour in enumerate(contours):
            if len(contour) > 0 and model.names[int(result.boxes.cls[i])] == 'nut':
                box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.int32)))
                nut_center = np.mean(box, axis=0)

        # 2. 모든 프렛 데이터 수집 (Box -> Smart Sort -> Top/Bot 분리)
        if nut_center is not None:
            for i, contour in enumerate(contours):
                if len(contour) > 0 and model.names[int(result.boxes.cls[i])] == 'fret_zone':
                    cnt = contour.astype(np.int32)
                    
                    # Box 사용 (진동 무시를 위해 Box 중심점만 쓰거나, 꼭짓점 사용)
                    # 여기서는 Polygon 대신 Box를 써도 충분함 (Regression이 알아서 평균을 내주므로)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect).astype(np.float32)
                    
                    # 정렬하여 Top/Bot 구분
                    sorted_pts = smart_sort_corners(box, nut_center)
                    # [Near_Top, Far_Top, Far_Bot, Near_Bot] 순서임
                    # Top Points: Near_Top, Far_Top
                    # Bot Points: Near_Bot, Far_Bot
                    
                    all_top_points.append(sorted_pts[0])
                    all_top_points.append(sorted_pts[1])
                    all_bot_points.append(sorted_pts[2])
                    all_bot_points.append(sorted_pts[3])
                    
                    all_x_coords.extend([pt[0] for pt in sorted_pts])

    # ---------------------------------------------------------
    # 3. Dual Regression (위/아래 독립 계산)
    # ---------------------------------------------------------
    if len(all_top_points) >= 6: # 점이 충분히 모였을 때
        # (A) Top Line Fitting
        pts_top_arr = np.array(all_top_points, dtype=np.int32)
        [vx_t, vy_t, x0_t, y0_t] = cv2.fitLine(pts_top_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        curr_top_params = np.array([vx_t, vy_t, x0_t, y0_t]).flatten()
        
        # (B) Bot Line Fitting
        pts_bot_arr = np.array(all_bot_points, dtype=np.int32)
        [vx_b, vy_b, x0_b, y0_b] = cv2.fitLine(pts_bot_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        curr_bot_params = np.array([vx_b, vy_b, x0_b, y0_b]).flatten()
        
        # (C) 안정화 (EMA)
        if prev_top_line is None:
            prev_top_line = curr_top_params
            prev_bot_line = curr_bot_params
        else:
            prev_top_line = (curr_top_params * alpha) + (prev_top_line * (1 - alpha))
            prev_bot_line = (curr_bot_params * alpha) + (prev_bot_line * (1 - alpha))

        # -----------------------------------------------------
        # 4. 선 그리기 (내분점 연결)
        # -----------------------------------------------------
        # 그리기 범위 설정 (X좌표 기준 Min/Max)
        x_min = min(all_x_coords) - 50
        x_max = max(all_x_coords) + 50
        
        # Top Line 함수: y = y0 + (x - x0) * (vy / vx)
        # (vx가 0일 경우 예외처리 필요하지만 기타 넥은 수직이 아니라고 가정)
        
        def get_point_on_line(params, x):
            vx, vy, x0, y0 = params
            if abs(vx) < 1e-3: return np.array([x0, y0]) # 수직선 방어
            t = (x - x0) / vx
            y = y0 + t * vy
            return np.array([x, y])

        # 시작점(Head쪽) 계산
        p_top_start = get_point_on_line(prev_top_line, x_min)
        p_bot_start = get_point_on_line(prev_bot_line, x_min)
        
        # 끝점(Body쪽) 계산
        p_top_end = get_point_on_line(prev_top_line, x_max)
        p_bot_end = get_point_on_line(prev_bot_line, x_max)

        # 4등분 선 긋기
        for k in range(1, 4):
            ratio = k / 4.0
            
            # 시작점 내분 (Head)
            p_start = p_top_start * (1 - ratio) + p_bot_start * ratio
            # 끝점 내분 (Body)
            p_end = p_top_end * (1 - ratio) + p_bot_end * ratio
            
            cv2.line(frame, tuple(p_start.astype(int)), tuple(p_end.astype(int)), (0, 255, 128), 2)
            
        # (디버깅) 위/아래 경계선 그리기 (파란색 얇게)
        # cv2.line(frame, tuple(p_top_start.astype(int)), tuple(p_top_end.astype(int)), (255, 0, 0), 1)
        # cv2.line(frame, tuple(p_bot_start.astype(int)), tuple(p_bot_end.astype(int)), (255, 0, 0), 1)

    alpha_blend = 0.4
    frame = cv2.addWeighted(overlay, alpha_blend, frame, 1 - alpha_blend, 0)
    
    cv2.imshow('Dual Regression Result', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("▶ 완료! 넥의 너비 변화(Taper)를 반영하여 선이 자연스럽게 직선으로 보일 것입니다.")