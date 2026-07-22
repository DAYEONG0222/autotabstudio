import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 1) 모델 로드
# =========================
fret_model = YOLO(os.path.join(BASE_DIR, "weights.pt"))
hand_model = YOLO(os.path.join(BASE_DIR, "weights_hands.pt"))

# =========================
# 2) MediaPipe (Fast + Strong) 이중화
# =========================
mp_hands = mp.solutions.hands

hands_fast = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35
)

hands_strong = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

# =========================
# 3) 비디오 로드
# =========================
cap = cv2.VideoCapture(os.path.join(BASE_DIR, "test_video6.mp4"))
width, height = int(cap.get(3)), int(cap.get(4))

fps_meta = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS =", fps_meta)

# =========================
# 4) 화면 설정 (안 잘림 + X로 닫으면 종료)
# =========================
WIN_NAME = "Bass Autotab - Length Stabilized"
TARGET_W, TARGET_H = 1280, 720
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, TARGET_W, TARGET_H)

# =========================
# 5) UI 설정
# =========================
TEXT_OFFSET_Y = 22
TEXT_OFFSET_X = 8
FINGERTIP_RADIUS = 4

STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
STR_THICKNESS, FRET_THICKNESS = 2, 2
ALPHA = 0.5

# =========================
# 6) 파라미터 / 상태 변수
# =========================
fingertips = [8, 12, 16, 20]
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

prev_top_line, prev_bot_line = None, None
prev_x_min, prev_x_max = None, None

# --- Hybrid tracking 상태 ---
stable_box = None
BOX_EMA_ALPHA = 0.35

tracked_tips = {}     # tip_id -> np.array([x,y])
smoothed_tips = {}    # tip_id -> np.array([x,y])
TIP_EMA_ALPHA = 0.5

prev_gray = None
miss_count = 999

FLOW_MAX_MISS = 120   # ~4초 @30fps
FLOW_JUMP_MAX = 60

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# --- B: YOLO 실패시 full-frame MP 재시드 ---
NOBOX_LIMIT = 10
no_box_count = 0
RESEED_W = 640
RESEED_PAD = 90

# --- C: 비대칭 ROI 패딩(넥 방향 +x 쪽이 더 넓게) ---
# 기본(평상시) 패딩
ROI_PAD_L, ROI_PAD_R = 70, 220
ROI_PAD_T, ROI_PAD_B = 110, 110

# 실패 시(더 넓게)
ROI_PAD_L2, ROI_PAD_R2 = 110, 320
ROI_PAD_T2, ROI_PAD_B2 = 160, 160

# --- C: 블러 감지 + 조건부 전처리 ---
BLUR_VAR_THRESHOLD = 55.0  # 낮을수록 더 흐림(영상마다 35~90 범위 튜닝)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

# --- C: 게이팅(오탐/튐 방지) ---
GATE_BOX_PAD = 60          # stable_box 주변 허용 범위
GATE_STRING_MARGIN = 80    # 스트링 영역(상/하 라인) 바깥 허용
GATE_X_MARGIN = 140        # prev_x_min/max 바깥 허용

# =========================
# 유틸 함수
# =========================
def ema_vec(prev, curr, a):
    return curr * a + prev * (1 - a)

def smart_sort_and_label_v6(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair, far_pair = pts[sorted_indices[:2]], pts[sorted_indices[2:]]
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]

    poly_pts = np.array([near_pair[0], far_pair[0], far_pair[1], near_pair[1]], dtype="float32")
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

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def make_padded_roi_asym(frame, box, pad_l, pad_r, pad_t, pad_b, w, h):
    x1, y1, x2, y2 = box
    x1p = max(0, x1 - pad_l)
    y1p = max(0, y1 - pad_t)
    x2p = min(w,  x2 + pad_r)
    y2p = min(h,  y2 + pad_b)
    roi = frame[y1p:y2p, x1p:x2p]
    return roi, x1p, y1p, x2p, y2p

def blur_variance(gray_roi):
    # Laplacian variance: 낮을수록 blur
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var()) if gray_roi.size else 1e9

def preprocess_roi_if_blurry(roi_bgr):
    """
    ROI가 blur하면(라플라시안 분산 낮으면) CLAHE + 약한 샤픈 적용.
    """
    if roi_bgr.size == 0:
        return roi_bgr

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    v = blur_variance(gray)
    if v >= BLUR_VAR_THRESHOLD:
        return roi_bgr

    # CLAHE on L channel (LAB)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # mild unsharp mask
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
    return out

def select_hand_box_flexible(hand_results, names):
    """
    left_hand 우선. 없으면 'hand' 포함 클래스 중 가장 큰 박스.
    """
    if hand_results.boxes is None:
        return None

    best = None
    best_area = -1

    # 1) left_hand 우선
    for b in hand_results.boxes:
        cls_name = names[int(b.cls[0])]
        if cls_name != "left_hand":
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    if best is not None:
        return best

    # 2) 없으면 hand류 중 최대
    for b in hand_results.boxes:
        cls_name = names[int(b.cls[0])]
        if "hand" not in cls_name.lower():
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    return best

def reseed_stable_box_by_fullframe_mp(frame_bgr, w, h, mp_instance):
    """
    full-frame(다운스케일)에서 MP로 손을 찾고, 랜드마크 bbox로 stable_box 재시드.
    """
    if w <= 0 or h <= 0:
        return None

    scale = RESEED_W / float(w)
    if scale >= 1.0:
        small = frame_bgr
        scale = 1.0
    else:
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    res = mp_instance.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None

    lms = res.multi_hand_landmarks[0]
    xs = [lm.x for lm in lms.landmark]
    ys = [lm.y for lm in lms.landmark]

    x_min = int(max(0, min(xs)) * small.shape[1])
    x_max = int(min(1, max(xs)) * small.shape[1])
    y_min = int(max(0, min(ys)) * small.shape[0])
    y_max = int(min(1, max(ys)) * small.shape[0])

    # back to original scale
    x_min = int(x_min / scale)
    x_max = int(x_max / scale)
    y_min = int(y_min / scale)
    y_max = int(y_max / scale)

    x1 = x_min - RESEED_PAD
    y1 = y_min - RESEED_PAD
    x2 = x_max + RESEED_PAD
    y2 = y_max + RESEED_PAD

    return clamp_box(x1, y1, x2, y2, w, h)

def point_in_string_band(x, y, top_line, bot_line, margin):
    """
    x에서 top/bot 라인 사이 + 여유(margin) 안에 y가 있는지 확인
    """
    yt = get_y_on_line(top_line, x)
    yb = get_y_on_line(bot_line, x)
    y_min = min(yt, yb) - margin
    y_max = max(yt, yb) + margin
    return (y >= y_min) and (y <= y_max)

def is_point_reasonable(x, y, w, h, stable_box, top_line, bot_line, x_min, x_max):
    """
    게이팅: 점이 말이 되는 위치인지 검사.
    """
    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    if stable_box is not None:
        sx1, sy1, sx2, sy2 = stable_box
        sx1 -= GATE_BOX_PAD; sy1 -= GATE_BOX_PAD
        sx2 += GATE_BOX_PAD; sy2 += GATE_BOX_PAD
        if not (sx1 <= x <= sx2 and sy1 <= y <= sy2):
            return False

    if top_line is not None and bot_line is not None:
        if not point_in_string_band(x, y, top_line, bot_line, GATE_STRING_MARGIN):
            return False

    if x_min is not None and x_max is not None:
        if not (x_min - GATE_X_MARGIN <= x <= x_max + GATE_X_MARGIN):
            return False

    return True

# =========================
# 메인 루프
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # (A) FRET YOLO (원본 유지)
    # -------------------------
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

    # --- [데이터 필터링 및 길이 안정화] ---
    all_top_pts, all_bot_pts, all_x = [], [], []
    if len(fret_data) >= 3:
        widths = [np.linalg.norm(f['tops'][0] - f['bots'][0]) for f in fret_data]

        for idx, f in enumerate(fret_data):
            is_valid = True
            if idx > 5:
                avg_prev_width = np.mean(widths[idx-3:idx])
                if widths[idx] < avg_prev_width * 0.94:
                    is_valid = False

            if is_valid:
                all_top_pts.extend(f['tops'])
                all_bot_pts.extend(f['bots'])

            all_x.extend([pt[0] for pt in f['poly']])

    if len(all_top_pts) >= 4:
        line_t = cv2.fitLine(np.array(all_top_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.001, 0.001).flatten()
        line_b = cv2.fitLine(np.array(all_bot_pts, dtype=np.int32), cv2.DIST_HUBER, 0, 0.001, 0.001).flatten()

        if prev_top_line is None:
            prev_top_line, prev_bot_line = line_t, line_b
        else:
            prev_top_line = (line_t * ALPHA) + (prev_top_line * (1 - ALPHA))
            prev_bot_line = (line_b * ALPHA) + (prev_bot_line * (1 - ALPHA))

    # --- [길이 스무딩] ---
    if all_x:
        curr_x_min, curr_x_max = min(all_x) - 50, max(all_x) + 50
        if prev_x_min is None:
            prev_x_min, prev_x_max = curr_x_min, curr_x_max
        else:
            prev_x_min = curr_x_min * 0.1 + prev_x_min * 0.9
            prev_x_max = curr_x_max * 0.1 + prev_x_max * 0.9

    # --- [시각화: 스트링] ---
    if prev_top_line is not None and prev_x_min is not None:
        for r in string_ratios:
            p1 = (int(prev_x_min),
                  int(get_y_on_line(prev_top_line, prev_x_min) * (1 - r) + get_y_on_line(prev_bot_line, prev_x_min) * r))
            p2 = (int(prev_x_max),
                  int(get_y_on_line(prev_top_line, prev_x_max) * (1 - r) + get_y_on_line(prev_bot_line, prev_x_max) * r))
            cv2.line(display_frame, p1, p2, STR_COLOR, STR_THICKNESS, cv2.LINE_AA)

    # --- [프렛 시각화] ---
    for idx, f in enumerate(fret_data):
        cv2.polylines(display_frame, [f['poly'].astype(np.int32)], True, FRET_COLOR, FRET_THICKNESS, cv2.LINE_AA)
        cv2.putText(display_frame, f"F{idx+1}", tuple(f['poly'][0].astype(int) + [0, -10]), 0, 0.4, FRET_COLOR, 1)

    # -------------------------
    # (B/C-pro) HAND: YOLO + MP(Fast/Strong) + Blur preprocess + Flow + Gating + Reseed
    # -------------------------

    # 1) hand YOLO (유연 선택 + 낮은 conf)
    hand_results = hand_model.predict(frame, conf=0.18, verbose=False)[0]
    yolo_box = select_hand_box_flexible(hand_results, hand_model.names)

    if yolo_box is None:
        no_box_count += 1
    else:
        no_box_count = 0

    # 2) stable_box EMA 업데이트
    if yolo_box is not None:
        if stable_box is None:
            stable_box = yolo_box
        else:
            prev = np.array(stable_box, dtype=np.float32)
            curr = np.array(yolo_box, dtype=np.float32)
            stable_box = tuple(ema_vec(prev, curr, BOX_EMA_ALPHA).tolist())

    # 3) YOLO가 연속 실패하면: full-frame MP로 재시드 (Strong)
    if yolo_box is None and no_box_count >= NOBOX_LIMIT:
        new_box = reseed_stable_box_by_fullframe_mp(frame, width, height, hands_strong)
        if new_box is not None:
            stable_box = new_box
            no_box_count = 0

    # 4) MediaPipe 시도: stable_box 기준 비대칭 ROI + blur면 전처리
    mp_ok = False
    mp_tip_points = {}

    if stable_box is not None:
        sx1, sy1, sx2, sy2 = clamp_box(*stable_box, width, height)

        # 1차: Fast + 기본 비대칭 패딩
        roi, ox1, oy1, ox2, oy2 = make_padded_roi_asym(
            frame, (sx1, sy1, sx2, sy2),
            ROI_PAD_L, ROI_PAD_R, ROI_PAD_T, ROI_PAD_B,
            width, height
        )
        if roi.size != 0:
            roi_in = preprocess_roi_if_blurry(roi)
            res = hands_fast.process(cv2.cvtColor(roi_in, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0]
                for tip in fingertips:
                    cx = int(lms.landmark[tip].x * roi.shape[1] + ox1)
                    cy = int(lms.landmark[tip].y * roi.shape[0] + oy1)
                    # 게이팅
                    if is_point_reasonable(cx, cy, width, height, stable_box, prev_top_line, prev_bot_line, prev_x_min, prev_x_max):
                        mp_tip_points[tip] = np.array([cx, cy], dtype=np.float32)
                if len(mp_tip_points) > 0:
                    mp_ok = True

        # 2차: Strong + 더 넓은 비대칭 패딩 (실패 시)
        if not mp_ok:
            roi2, ox1b, oy1b, ox2b, oy2b = make_padded_roi_asym(
                frame, (sx1, sy1, sx2, sy2),
                ROI_PAD_L2, ROI_PAD_R2, ROI_PAD_T2, ROI_PAD_B2,
                width, height
            )
            if roi2.size != 0:
                roi2_in = preprocess_roi_if_blurry(roi2)
                res2 = hands_strong.process(cv2.cvtColor(roi2_in, cv2.COLOR_BGR2RGB))
                if res2.multi_hand_landmarks:
                    lms = res2.multi_hand_landmarks[0]
                    mp_tip_points = {}
                    for tip in fingertips:
                        cx = int(lms.landmark[tip].x * roi2.shape[1] + ox1b)
                        cy = int(lms.landmark[tip].y * roi2.shape[0] + oy1b)
                        if is_point_reasonable(cx, cy, width, height, stable_box, prev_top_line, prev_bot_line, prev_x_min, prev_x_max):
                            mp_tip_points[tip] = np.array([cx, cy], dtype=np.float32)
                    if len(mp_tip_points) > 0:
                        mp_ok = True

    # 5) MP 성공 -> 갱신
    if mp_ok:
        miss_count = 0
        for tip, pt in mp_tip_points.items():
            tracked_tips[tip] = pt.copy()
            if tip not in smoothed_tips:
                smoothed_tips[tip] = pt.copy()
            else:
                smoothed_tips[tip] = ema_vec(smoothed_tips[tip], pt, TIP_EMA_ALPHA)

    # 6) MP 실패 -> Flow로 버티기 (게이팅 포함)
    else:
        miss_count += 1
        if prev_gray is not None and len(tracked_tips) > 0 and miss_count <= FLOW_MAX_MISS:
            prev_pts = np.array([tracked_tips[t] for t in fingertips if t in tracked_tips], dtype=np.float32).reshape(-1, 1, 2)
            tip_ids = [t for t in fingertips if t in tracked_tips]

            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)

            if next_pts is not None and st is not None:
                for i, tip in enumerate(tip_ids):
                    if st[i, 0] != 1:
                        continue

                    nx, ny = next_pts[i, 0]
                    nx = float(np.clip(nx, 0, width - 1))
                    ny = float(np.clip(ny, 0, height - 1))

                    prev_pt = tracked_tips[tip]
                    jump = float(np.linalg.norm(np.array([nx, ny], dtype=np.float32) - prev_pt))
                    if jump > FLOW_JUMP_MAX:
                        continue

                    # 게이팅: 말이 안 되는 위치면 업데이트 무시
                    if not is_point_reasonable(int(nx), int(ny), width, height, stable_box, prev_top_line, prev_bot_line, prev_x_min, prev_x_max):
                        continue

                    tracked_tips[tip] = np.array([nx, ny], dtype=np.float32)
                    if tip not in smoothed_tips:
                        smoothed_tips[tip] = tracked_tips[tip].copy()
                    else:
                        smoothed_tips[tip] = ema_vec(smoothed_tips[tip], tracked_tips[tip], TIP_EMA_ALPHA)

        else:
            if miss_count > FLOW_MAX_MISS:
                tracked_tips.clear()
                smoothed_tips.clear()

    # 7) smoothed_tips로 스트링/프렛 판정 + UI (텍스트는 점 아래, 점 작게)
    if prev_top_line is not None and len(smoothed_tips) > 0:
        for tip in fingertips:
            if tip not in smoothed_tips:
                continue

            cx, cy = smoothed_tips[tip]
            cx_i, cy_i = int(cx), int(cy)

            # 마지막 한 번 더 게이팅(안전)
            if not is_point_reasonable(cx_i, cy_i, width, height, stable_box, prev_top_line, prev_bot_line, prev_x_min, prev_x_max):
                continue

            best_s, min_dist = -1, float('inf')
            for s_idx, r in enumerate(string_ratios):
                target_y = get_y_on_line(prev_top_line, cx_i) * (1 - r) + get_y_on_line(prev_bot_line, cx_i) * r
                dist = abs(cy_i - target_y)
                if dist < min_dist:
                    min_dist, best_s = dist, 4 - s_idx

            current_f = 0
            for f_num, f in enumerate(fret_data):
                if cv2.pointPolygonTest(f['poly'], (cx_i, cy_i), False) >= 0:
                    current_f = f_num + 1
                    break

            if min_dist < 35:
                txt = f"S{best_s} F{current_f}"
                (tw, th), _ = cv2.getTextSize(txt, 0, 0.5, 2)

                # 🔴 빨간 점(작게)
                cv2.circle(display_frame, (cx_i, cy_i), FINGERTIP_RADIUS, (0, 0, 255), -1)

                # 🟨 텍스트(아래)
                text_x = cx_i + TEXT_OFFSET_X
                text_y = min(cy_i + TEXT_OFFSET_Y, height - 10)

                cv2.rectangle(
                    display_frame,
                    (text_x - 2, text_y - th - 2),
                    (text_x + tw + 2, text_y + 4),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(display_frame, txt, (text_x, text_y), 0, 0.5, (0, 255, 255), 2)

    prev_gray = gray

    # =========================
    # 출력(잘림 방지) + 닫기 버튼 종료
    # =========================
    h, w = display_frame.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h, 1.0)
    show = display_frame
    if scale < 1.0:
        show = cv2.resize(display_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow(WIN_NAME, show)

    if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
