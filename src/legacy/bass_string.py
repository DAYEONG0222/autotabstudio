from ultralytics import YOLO
import cv2
import numpy as np

# 1. 모델 및 비디오 설정
model = YOLO(r'C:\Autotab_Studio\weights.pt')
video_path = r'C:\Autotab_Studio\test_video2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Bass_4String_Guide.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

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

# 🛠️ 베이스 전용 설정
prev_top_line = None
prev_bot_line = None
alpha = 0.5            # 안정화 계수
line_thickness = 1     # 가시성 높은 굵기
line_color = (50, 255, 50)  # 네온 그린

print("▶ 베이스 4현 전용 가이드 (1:2:2:2:1 비율) 렌더링 중...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    overlay = frame.copy()
    results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
    result = results[0]

    all_top_points, all_bot_points, all_x_coords = [], [], []
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
        # Dual Regression 피팅
        pts_top_arr = np.array(all_top_points, dtype=np.int32)
        [vx_t, vy_t, x0_t, y0_t] = cv2.fitLine(pts_top_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        curr_top_params = np.array([vx_t, vy_t, x0_t, y0_t]).flatten()
        
        pts_bot_arr = np.array(all_bot_points, dtype=np.int32)
        [vx_b, vy_b, x0_b, y0_b] = cv2.fitLine(pts_bot_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        curr_bot_params = np.array([vx_b, vy_b, x0_b, y0_b]).flatten()
        
        # EMA 안정화
        if prev_top_line is None:
            prev_top_line, prev_bot_line = curr_top_params, curr_bot_params
        else:
            prev_top_line = (curr_top_params * alpha) + (prev_top_line * (1 - alpha))
            prev_bot_line = (curr_bot_params * alpha) + (prev_bot_line * (1 - alpha))

        # 라인 좌표 계산
        x_min, x_max = min(all_x_coords) - 50, max(all_x_coords) + 50
        def get_point_on_line(params, x):
            vx, vy, x0, y0 = params
            return np.array([x, y0 + ((x - x0) / vx) * vy])

        p_top_start, p_bot_start = get_point_on_line(prev_top_line, x_min), get_point_on_line(prev_bot_line, x_min)
        p_top_end, p_bot_end = get_point_on_line(prev_top_line, x_max), get_point_on_line(prev_bot_line, x_max)

        # 8등분 기반 홀수선(1,3,5,7) 그리기
        for k in [0.9, 2.85, 4.85, 6.8]:
            ratio = k / 8.0
            p_start = p_top_start * (1 - ratio) + p_bot_start * ratio
            p_end = p_top_end * (1 - ratio) + p_bot_end * ratio
            cv2.line(frame, tuple(p_start.astype(int)), tuple(p_end.astype(int)), 
                     line_color, line_thickness, cv2.LINE_AA)

    # 투명도 합성 및 출력
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.imshow('Bass 4-String Guide', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()