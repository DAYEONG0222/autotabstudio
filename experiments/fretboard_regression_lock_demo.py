from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r'C:\Autotab_Studio\weights.pt')
video_path = r'C:\Autotab_Studio\test_video2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_dual_auto.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# 광학 흐름 파라미터
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 상태 변수
is_tracking = False
p0 = None       # 추적할 4개의 가상 앵커 포인트
old_gray = None # 이전 프레임 이미지

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
        cross_val = axis[0] * pt_vec[1] - axis[1] * pt_vec[0]
        return cross_val < 0 

    if is_above_axis(near_pair[0], nut_center, axis_vec):
        near_top, near_bot = near_pair[0], near_pair[1]
    else:
        near_top, near_bot = near_pair[1], near_pair[0]
        
    if is_above_axis(far_pair[0], nut_center, axis_vec):
        far_top, far_bot = far_pair[0], far_pair[1]
    else:
        far_top, far_bot = far_pair[1], far_pair[0]

    return np.array([near_top, far_top, far_bot, near_bot], dtype="float32")

print("▶ Dual Regression + Auto Lock: 계산이 완료되면 즉시 트래킹으로 전환합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()
    
    # =======================================================
    # 모드 1: 트래킹 (YOLO OFF, Dual Regression 결과만 추적)
    # =======================================================
    if is_tracking and p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None and len(p1) == 4:
            s_top, s_bot, e_top, e_bot = p1.reshape(4, 2)
            
            for k in range(1, 4):
                ratio = k / 4.0
                p_start = s_top * (1 - ratio) + s_bot * ratio
                p_end = e_top * (1 - ratio) + e_bot * ratio
                
                # 연두색 선
                cv2.line(display_frame, tuple(p_start.astype(int)), tuple(p_end.astype(int)), (0, 255, 128), 2)
            
            cv2.putText(display_frame, "MODE: TRACKING (Fast)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
            
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)
        else:
            print("❌ 추적 실패. 다시 탐색합니다.")
            is_tracking = False
            p0 = None

    # =======================================================
    # 모드 2: 탐색 (YOLO ON + Dual Regression 계산)
    # =======================================================
    else:
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

        # Dual Regression 계산
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

            x_min = min(all_x_coords)
            x_max = max(all_x_coords)
            
            p_st = get_point(vx_t, vy_t, x0_t, y0_t, x_min)
            p_sb = get_point(vx_b, vy_b, x0_b, y0_b, x_min)
            p_et = get_point(vx_t, vy_t, x0_t, y0_t, x_max)
            p_eb = get_point(vx_b, vy_b, x0_b, y0_b, x_max)
            
            # 🚀 [핵심 변경] 계산되자마자 즉시 Lock & Track
            temp_anchors = np.array([p_st, p_sb, p_et, p_eb], dtype=np.float32)
            p0 = temp_anchors.reshape(-1, 1, 2)
            old_gray = frame_gray.copy()
            is_tracking = True
            print(">>> 🚀 Auto-Lock! 트래킹 모드 진입 (Fast)")

        else:
            cv2.putText(display_frame, "Calculating Regression...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Auto Dual Regression', display_frame)
    out.write(display_frame) # 원본 frame 대신 display_frame 저장
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()