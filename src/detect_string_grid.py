import cv2
import numpy as np
from ultralytics import YOLO
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# YOLO Fret 모델만 사용
# -----------------------------
fret_model = YOLO(os.path.join(PROJECT_ROOT, "data", "weights", "weights.pt"))

# -----------------------------
# 내부 상태 저장용 변수 (스무딩 유지)
# -----------------------------
prev_top_line = None
prev_bot_line = None
prev_x_min = None
prev_x_max = None

ALPHA = 0.5
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]


# -------------------------------------------------
# 기존 보조 함수들 그대로 유지
# -------------------------------------------------
def smart_sort_and_label_v6(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair, far_pair = pts[sorted_indices[:2]], pts[sorted_indices[2:]]
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]

    poly_pts = np.array(
        [near_pair[0], far_pair[0], far_pair[1], near_pair[1]],
        dtype="float32"
    )

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


# -------------------------------------------------
# 🔥 핵심 함수: 스트링 라인 계산
# -------------------------------------------------
def detect_strings(frame):
    """
    입력:
        frame (BGR)

    출력:
        {
            "string_lines": [(x1,y1,x2,y2), ...],
            "top_line": prev_top_line,
            "bot_line": prev_bot_line,
            "x_min": prev_x_min,
            "x_max": prev_x_max
        }
    """

    global prev_top_line, prev_bot_line
    global prev_x_min, prev_x_max

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

    all_top_pts, all_bot_pts, all_x = [], [], []

    if len(fret_data) >= 3:
        for f in fret_data:
            all_top_pts.extend(f['tops'])
            all_bot_pts.extend(f['bots'])
            all_x.extend([pt[0] for pt in f['poly']])

    if len(all_top_pts) >= 4:
        line_t = cv2.fitLine(np.array(all_top_pts, dtype=np.int32),
                             cv2.DIST_HUBER, 0, 0.001, 0.001).flatten()
        line_b = cv2.fitLine(np.array(all_bot_pts, dtype=np.int32),
                             cv2.DIST_HUBER, 0, 0.001, 0.001).flatten()

        if prev_top_line is None:
            prev_top_line, prev_bot_line = line_t, line_b
        else:
            prev_top_line = (line_t * ALPHA) + (prev_top_line * (1 - ALPHA))
            prev_bot_line = (line_b * ALPHA) + (prev_bot_line * (1 - ALPHA))

    if all_x:
        curr_x_min, curr_x_max = min(all_x) - 50, max(all_x) + 50
        if prev_x_min is None:
            prev_x_min, prev_x_max = curr_x_min, curr_x_max
        else:
            prev_x_min = curr_x_min * 0.1 + prev_x_min * 0.9
            prev_x_max = curr_x_max * 0.1 + prev_x_max * 0.9

    string_lines = []

    if prev_top_line is not None and prev_x_min is not None:
        for r in string_ratios:
            x1 = int(prev_x_min)
            x2 = int(prev_x_max)

            y1 = int(
                get_y_on_line(prev_top_line, x1) * (1 - r) +
                get_y_on_line(prev_bot_line, x1) * r
            )

            y2 = int(
                get_y_on_line(prev_top_line, x2) * (1 - r) +
                get_y_on_line(prev_bot_line, x2) * r
            )

            string_lines.append((x1, y1, x2, y2))

    return {
        "string_lines": string_lines,
        "top_line": prev_top_line,
        "bot_line": prev_bot_line,
        "x_min": prev_x_min,
        "x_max": prev_x_max
    }


if __name__ == "__main__":
    print("new3_grid.py is now a module. Use detect_strings(frame).")
