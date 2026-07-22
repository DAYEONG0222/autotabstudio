from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

# 경로 지정 필요
model = YOLO('../../data/models/weights_bass.pt')
video_path = '../../data/input/drowning.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../../data/output/output_ransac.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


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
        # NumPy 2.0 호환
        axis_3d = np.append(axis, 0)
        pt_vec_3d = np.append(pt_vec, 0)
        return np.cross(axis_3d, pt_vec_3d)[2] < 0

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
# 🛠️ RANSAC 회귀 함수
# -----------------------------------------------------------
def fit_ransac_line(points):
    """RANSAC을 사용한 강건한 선형 회귀"""
    if len(points) < 6:
        return None

    x_coords = points[:, 0].reshape(-1, 1)
    y_coords = points[:, 1]

    ransac = RANSACRegressor(random_state=42, residual_threshold=5.0)
    ransac.fit(x_coords, y_coords)

    return ransac


def predict_y(ransac_model, x):
    """RANSAC 모델로 y값 예측"""
    if ransac_model is None:
        return 0
    return ransac_model.predict(np.array([[x]]))[0]


# -----------------------------------------------------------
# 🛠️ 안정화 변수
# -----------------------------------------------------------
prev_top_ransac = None
prev_bot_ransac = None
alpha = 0.5

print("▶ RANSAC Regression 적용: 이상치에 강건한 회귀로 안정적인 선 피팅")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    overlay = frame.copy()
    results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
    result = results[0]

    all_top_points = []  # 1번줄 쪽 점들 모음
    all_bot_points = []  # 4번줄 쪽 점들 모음
    all_x_coords = []  # 그리기 범위 결정을 위한 X좌표 모음

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

                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect).astype(np.float32)

                    # 정렬하여 Top/Bot 구분
                    sorted_pts = smart_sort_corners(box, nut_center)

                    all_top_points.append(sorted_pts[0])
                    all_top_points.append(sorted_pts[1])
                    all_bot_points.append(sorted_pts[2])
                    all_bot_points.append(sorted_pts[3])

                    all_x_coords.extend([pt[0] for pt in sorted_pts])

    # ---------------------------------------------------------
    # 3. RANSAC Regression
    # ---------------------------------------------------------
    if len(all_top_points) >= 6:
        # (A) Top Line RANSAC Fitting
        pts_top_arr = np.array(all_top_points, dtype=np.float32)
        ransac_top = fit_ransac_line(pts_top_arr)

        # (B) Bot Line RANSAC Fitting
        pts_bot_arr = np.array(all_bot_points, dtype=np.float32)
        ransac_bot = fit_ransac_line(pts_bot_arr)

        # (C) 모델 업데이트 (첫 프레임 또는 새 모델 사용)
        if prev_top_ransac is None:
            prev_top_ransac = ransac_top
            prev_bot_ransac = ransac_bot
        else:
            # RANSAC 모델은 직접 EMA 적용이 어려우므로 새 모델 사용
            prev_top_ransac = ransac_top
            prev_bot_ransac = ransac_bot

        # -----------------------------------------------------
        # 4. 선 그리기
        # -----------------------------------------------------
        x_min = min(all_x_coords) - 50
        x_max = max(all_x_coords) + 50

        # 시작점 계산
        p_top_start = np.array([x_min, predict_y(prev_top_ransac, x_min)])
        p_bot_start = np.array([x_min, predict_y(prev_bot_ransac, x_min)])

        # 끝점 계산
        p_top_end = np.array([x_max, predict_y(prev_top_ransac, x_max)])
        p_bot_end = np.array([x_max, predict_y(prev_bot_ransac, x_max)])

        # 4등분선 그리기
        for k in range(1, 4):
            ratio = k / 4.0

            # 시작점 내분 (Head)
            p_start = p_top_start * (1 - ratio) + p_bot_start * ratio
            # 끝점 내분 (Body)
            p_end = p_top_end * (1 - ratio) + p_bot_end * ratio

            cv2.line(frame, tuple(p_start.astype(int)), tuple(p_end.astype(int)), (0, 255, 128), 2)

    alpha_blend = 0.4
    frame = cv2.addWeighted(overlay, alpha_blend, frame, 1 - alpha_blend, 0)

    cv2.imshow('RANSAC Regression Result', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("▶ 완료! RANSAC 회귀로 이상치에 강건한 피팅 완료")