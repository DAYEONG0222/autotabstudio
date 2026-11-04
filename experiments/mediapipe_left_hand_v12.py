import cv2
import numpy as np
import torch

print(f"CUDA 가동 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")  # RTX 5070은 9.0 이상이 나와야 함
else:
    print("⚠️ 경고: GPU를 찾지 못해 CPU로 돌아가고 있습니다!")

import os
import time
import pandas as pd
import tkinter as tk
from ultralytics import YOLO

# [1] 모델 및 디바이스 초기화
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except:
    from mediapipe.python.solutions import hands as mp_hands

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Segmentation 모델 (손, 너트, 프렛 인식용)
seg_model = YOLO(
    r'data/weights/weights.pt'
).to(device)

hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# [2] 오디오 히트 데이터 로드 (venv-A에서 생성한 CSV)
HIT_FILE = "hits.csv"
onset_frames = []
if os.path.exists(HIT_FILE):
    try:
        df = pd.read_csv(HIT_FILE)
        onset_frames = df['frame'].tolist()
        print(f"{len(onset_frames)}개의 타점 프레임 동기화 완료")
    except:
        print("hits.csv 로드 실패. 실시간 보정 모드로 작동합니다.")

# (옵션이지만 추천) membership 빠르게
onset_frames_set = set(onset_frames)

# [3] 필터 및 기하학 유틸리티
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x_prev, self.dx_prev, self.t_prev = float(x0), float(dx0), float(t0)

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        a_d = (2 * np.pi * self.d_cutoff * t_e) / (2 * np.pi * self.d_cutoff * t_e + 1)
        dx_hat = a_d * ((x - self.x_prev) / t_e) + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * np.pi * cutoff * t_e) / (2 * np.pi * cutoff * t_e + 1)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.t_prev, self.dx_prev = x_hat, t, dx_hat
        return x_hat

class PointFilter:
    def __init__(self, t0, x0, y0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.reset(t0, x0, y0, min_cutoff, beta, d_cutoff)

    def reset(self, t, x, y, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.x_filter = OneEuroFilter(t, x, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.y_filter = OneEuroFilter(t, y, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def process(self, t, pt):
        return (
            self.x_filter.filter(t, pt[0]),
            self.y_filter.filter(t, pt[1])
        )

def get_y_on_line(params, x):
    if params is None or x is None:
        return 0
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

# [4] V28 핵심: 중력 없는 기하학적 재정렬 엔진
def refine_landmarks_v28(mp_lms, roi_offset, roi_shape, nut_center, hand_mask):
    if not mp_lms or nut_center is None:
        return {}
    tip_indices = [8, 12, 16, 20]
    valid_points = []

    for idx in tip_indices:
        lm = mp_lms.landmark[idx]
        tx, ty = int(lm.x * roi_shape[1] + roi_offset[0]), int(lm.y * roi_shape[0] + roi_offset[1])
        if 0 <= ty < hand_mask.shape[0] and 0 <= tx < hand_mask.shape[1]:
            if hand_mask[ty, tx] > 0:
                valid_points.append(np.array([tx, ty]))

    if len(valid_points) < 4:
        return {}

    valid_points.sort(key=lambda p: np.linalg.norm(p - nut_center))

    return {
        8:  (int(valid_points[0][0]), int(valid_points[0][1])),
        12: (int(valid_points[1][0]), int(valid_points[1][1])),
        16: (int(valid_points[2][0]), int(valid_points[2][1])),
        20: (int(valid_points[3][0]), int(valid_points[3][1]))
    }

# =========================
# [A] 중지(및 전체 손끝) 튐 억제: 마스크 스냅 + 점프 제한 + OneEuroFilter
# =========================
def snap_to_mask(pt, hand_mask, radius=18):
    """pt 주변 radius 안에서 hand_mask(>0) 픽셀 중 가장 가까운 점으로 스냅.
    못 찾으면 원점 반환."""
    x, y = int(pt[0]), int(pt[1])
    h, w = hand_mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return pt

    if hand_mask[y, x] > 0:
        return pt

    x0, x1 = max(0, x - radius), min(w - 1, x + radius)
    y0, y1 = max(0, y - radius), min(h - 1, y + radius)
    patch = hand_mask[y0:y1 + 1, x0:x1 + 1]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return pt

    dx = xs + x0 - x
    dy = ys + y0 - y
    i = int(np.argmin(dx * dx + dy * dy))
    return (int(xs[i] + x0), int(ys[i] + y0))

def limit_jump(prev_pt, new_pt, max_jump_px=45):
    """이전 좌표 대비 점프가 크면, 이동량을 max_jump_px로 제한."""
    if prev_pt is None:
        return new_pt
    px, py = prev_pt
    nx, ny = new_pt
    dx, dy = (nx - px), (ny - py)
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= max_jump_px:
        return new_pt
    if dist < 1e-6:
        return prev_pt
    s = max_jump_px / dist
    return (int(px + dx * s), int(py + dy * s))

# 손끝별 필터/이전값 상태
fingertips = [8, 12, 16, 20]
pt_filters = {tip: None for tip in fingertips}
prev_pts = {tip: None for tip in fingertips}

# 필터 파라미터(중지가 더 튀면 중지에만 더 강하게 줄 수도 있음)
# min_cutoff: 낮을수록 더 부드러움(지연↑), beta: 빠른 움직임 때 따라가게 하는 정도
FILTER_MIN_CUTOFF_DEFAULT = 1.3
FILTER_BETA_DEFAULT = 0.010
FILTER_D_CUTOFF = 1.0

# 중지(12) 전용 강화(원하면 여기만 더 조절)
FILTER_MIN_CUTOFF_MIDDLE = 1.6
FILTER_BETA_MIDDLE = 0.008

MAX_JUMP_DEFAULT = 45
MAX_JUMP_MIDDLE = 35
SNAP_RADIUS_DEFAULT = 18
SNAP_RADIUS_MIDDLE = 22

def stabilize_tip(t, tip_idx, raw_pt, hand_mask):
    """raw_pt -> (스냅) -> (점프 제한) -> (OneEuroFilter)"""
    if raw_pt is None:
        return None

    # tip별 파라미터
    if tip_idx == 12:
        max_jump = MAX_JUMP_MIDDLE
        snap_r = SNAP_RADIUS_MIDDLE
        min_cutoff = FILTER_MIN_CUTOFF_MIDDLE
        beta = FILTER_BETA_MIDDLE
    else:
        max_jump = MAX_JUMP_DEFAULT
        snap_r = SNAP_RADIUS_DEFAULT
        min_cutoff = FILTER_MIN_CUTOFF_DEFAULT
        beta = FILTER_BETA_DEFAULT

    # 1) 마스크 스냅
    pt1 = snap_to_mask(raw_pt, hand_mask, radius=snap_r)

    # 2) 점프 제한
    pt2 = limit_jump(prev_pts[tip_idx], pt1, max_jump_px=max_jump)

    # 3) OneEuroFilter
    if pt_filters[tip_idx] is None:
        pt_filters[tip_idx] = PointFilter(t, pt2[0], pt2[1], min_cutoff=min_cutoff, beta=beta, d_cutoff=FILTER_D_CUTOFF)
        pt3 = pt2
    else:
        # (필요하면 상황에 따라 파라미터를 바꾸고 싶을 때 reset도 가능)
        pt3f = pt_filters[tip_idx].process(t, pt2)
        pt3 = (int(pt3f[0]), int(pt3f[1]))

    prev_pts[tip_idx] = pt3
    return pt3

# [5] 메인 루프 실행부
cap = cv2.VideoCapture(
    r'data/videos/test_video6.mp4'
)
width, height = int(cap.get(3)), int(cap.get(4))

# =========================
# [추가] 결과 영상 저장 설정
# =========================
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = r'outputs/output_result.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"🎬 결과 영상 저장 시작: {output_path}")

STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

prev_t, prev_b, px_min, px_max, last_nut = None, None, None, None, None
frame_count = 0

# =========================
# 화면에 맞게 전체가 보이도록: 창 리사이즈 + 모니터 크기 기반 자동 축소
# =========================
WIN = 'Autotab V28.2 - Green Tracking Mode'
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

root = tk.Tk()
root.withdraw()
SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()
root.destroy()

MARGIN_W, MARGIN_H = 40, 120
MAX_W = max(200, SCREEN_W - MARGIN_W)
MAX_H = max(200, SCREEN_H - MARGIN_H)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_t = time.time()
    display_frame = frame.copy()

    # YOLO Segmentation 추론
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
                hand_detected = True
                hand_box = results.boxes.xyxy[i].cpu().numpy()
            elif cls == 'fret_zone' and last_nut is not None:
                poly, tops, bots, d = smart_sort_and_label_v6(
                    cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour.astype(np.int32)))),
                    last_nut
                )
                fret_data.append({'poly': poly, 'tops': tops, 'bots': bots, 'dist': d})

    # 지판 그리드 시각화
    fret_data.sort(key=lambda x: x['dist'])
    all_top, all_bot, all_x = [], [], []
    if len(fret_data) >= 3:
        for f in fret_data:
            all_top.extend(f['tops'])
            all_bot.extend(f['bots'])
            all_x.extend([pt[0] for pt in f['poly']])
            cv2.polylines(display_frame, [f['poly'].astype(np.int32)], True, FRET_COLOR, 2)

        if len(all_top) >= 4:
            lt = cv2.fitLine(np.array(all_top, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            lb = cv2.fitLine(np.array(all_bot, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            prev_t, prev_b = lt, lb
            px_min, px_max = min(all_x) - 50, max(all_x) + 50

    if prev_t is not None:
        for r in string_ratios:
            p1 = (int(px_min), int(get_y_on_line(prev_t, px_min) * (1 - r) + get_y_on_line(prev_b, px_min) * r))
            p2 = (int(px_max), int(get_y_on_line(prev_t, px_max) * (1 - r) + get_y_on_line(prev_b, px_max) * r))
            cv2.line(display_frame, p1, p2, STR_COLOR, 1)

    # 손가락 분석
    raw_coords = {}
    is_hit_frame = frame_count in onset_frames_set

    if hand_detected:
        x1, y1, x2, y2 = map(int, hand_box)
        pad = 30
        roi_x = max(0, x1 - pad)
        roi_y = max(0, y1 - pad)
        roi = frame[roi_y:min(height, y2 + pad), roi_x:min(width, x2 + pad)]

        mp_res = hands_mp.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if mp_res.multi_hand_landmarks:
            if is_hit_frame:
                raw_coords = refine_landmarks_v28(
                    mp_res.multi_hand_landmarks[0],
                    (roi_x, roi_y),
                    roi.shape,
                    last_nut,
                    hand_mask
                )
            else:
                for idx in fingertips:
                    lm = mp_res.multi_hand_landmarks[0].landmark[idx]
                    raw_coords[idx] = (
                        int(lm.x * roi.shape[1] + roi_x),
                        int(lm.y * roi.shape[0] + roi_y)
                    )

    # =========================
    # [A 적용] raw_coords -> stabilize -> final_coords
    # =========================
    final_coords = {}
    for tip in fingertips:
        if tip in raw_coords:
            stable = stabilize_tip(curr_t, tip, raw_coords[tip], hand_mask)
            if stable is not None:
                final_coords[tip] = stable

    # 최종 렌더링
    for tip, (cx, cy) in final_coords.items():
        bs, mdist = -1, float('inf')
        if prev_t is not None:
            for s_idx, r in enumerate(string_ratios):
                ty = get_y_on_line(prev_t, cx) * (1 - r) + get_y_on_line(prev_b, cx) * r
                d = abs(cy - ty)
                if d < mdist:
                    mdist, bs = d, 4 - s_idx

        cf = 0
        for f_num, f_obj in enumerate(fret_data):
            if cv2.pointPolygonTest(f_obj['poly'], (float(cx), float(cy)), False) >= 0:
                cf = f_num + 1
                break

        color = (0, 255, 255) if is_hit_frame else (0, 0, 255)
        size = 8 if is_hit_frame else 4
        cv2.circle(display_frame, (cx, cy), size, color, -1)

        if mdist < 45:
            label = f"S{bs} F{cf}" + (" [HIT]" if is_hit_frame else "")
            cv2.putText(display_frame, label, (cx + 10, cy), 0, 0.5, color, 2)

    # 원본 해상도 저장
    out.write(display_frame)

    # 화면 표시(자동 축소)
    h, w = display_frame.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    show = display_frame if scale == 1.0 else cv2.resize(display_frame, (int(w * scale), int(h * scale)))

    cv2.imshow(WIN, show)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 영상 저장 완료")