import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 모델 및 MediaPipe 초기화
fret_model = YOLO(os.path.join(BASE_DIR, "weights.pt"))
hand_model = YOLO(os.path.join(BASE_DIR, "weights_hands.pt"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

cap = cv2.VideoCapture(os.path.join(BASE_DIR, "test_video.mp4"))
width, height = int(cap.get(3)), int(cap.get(4))

# =========================
# 화면 표시 설정 (안 잘리게 + X로 닫으면 종료)
# =========================
WIN_NAME = "Bass Autotab - Length Stabilized"
TARGET_W, TARGET_H = 1280, 720
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, TARGET_W, TARGET_H)

# 시각화 설정
STR_COLOR, FRET_COLOR = (50, 255, 50), (255, 255, 100)
STR_THICKNESS, FRET_THICKNESS = 2, 2
ALPHA = 0.5

# 선과 길이를 위한 전역 변수
prev_top_line, prev_bot_line = None, None
prev_x_min, prev_x_max = None, None
fingertips = [8, 12, 16, 20]
string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

# =========================
# ✅ Hybrid: Hand ROI 안정화 + Optical Flow + fingertip smoothing
# =========================
PAD = 60

# (1) YOLO hand box EMA 안정화
BOX_EMA_ALPHA = 0.35      # 0.2~0.45 추천 (높을수록 더 빠르게 따라감)
stable_box = None         # (x1, y1, x2, y2) in full-frame coords

# (2) Optical Flow fallback
FLOW_MAX_MISS = 6         # MP가 끊겨도 최대 N프레임은 flow로 버팀
miss_count = 999
prev_gray = None

# tracked_tips: tip_id -> np.array([x,y]) (full-frame coords)
tracked_tips = {}
# smoothed_tips: tip_id -> np.array([x,y])
smoothed_tips = {}

TIP_EMA_ALPHA = 0.5       # 0.35~0.7 추천 (낮을수록 더 부드럽게)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

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

def make_padded_roi(frame, box, pad, w, h):
    x1, y1, x2, y2 = box
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w,  x2 + pad)
    y2p = min(h,  y2 + pad)
    roi = frame[y1p:y2p, x1p:x2p]
    return roi, x1p, y1p, x2p, y2p

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

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
            p1 = (int(prev_x_min), int(get_y_on_line(prev_top_line, prev_x_min)*(1-r) + get_y_on_line(prev_bot_line, prev_x_min)*r))
            p2 = (int(prev_x_max), int(get_y_on_line(prev_top_line, prev_x_max)*(1-r) + get_y_on_line(prev_bot_line, prev_x_max)*r))
            cv2.line(display_frame, p1, p2, STR_COLOR, STR_THICKNESS, cv2.LINE_AA)

    # --- [프렛 표시 유지] ---
    for idx, f in enumerate(fret_data):
        cv2.polylines(display_frame, [f['poly'].astype(np.int32)], True, FRET_COLOR, FRET_THICKNESS, cv2.LINE_AA)
        cv2.putText(display_frame, f"F{idx+1}", tuple(f['poly'][0].astype(int) + [0, -10]), 0, 0.4, FRET_COLOR, 1)

    # -------------------------
    # (B) HAND YOLO + Hybrid MP/Flow (여기만 강화)
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) YOLO로 left_hand box 얻기 (가능하면 1개만)
    hand_results = hand_model.predict(frame, conf=0.3, verbose=False)[0]
    yolo_box = None
    if hand_results.boxes is not None:
        for h_box in hand_results.boxes:
            if hand_model.names[int(h_box.cls[0])] != 'left_hand':
                continue
            x1, y1, x2, y2 = map(int, h_box.xyxy[0])
            yolo_box = (x1, y1, x2, y2)
            break

    # 2) stable_box 업데이트 (EMA)
    if yolo_box is not None:
        if stable_box is None:
            stable_box = yolo_box
        else:
            prev = np.array(stable_box, dtype=np.float32)
            curr = np.array(yolo_box, dtype=np.float32)
            blended = ema_vec(prev, curr, BOX_EMA_ALPHA)
            stable_box = tuple(blended.tolist())

    # 3) MediaPipe 시도 (stable_box 기준 ROI)
    mp_ok = False
    mp_tip_points = {}

    if stable_box is not None:
        x1, y1, x2, y2 = clamp_box(*stable_box, width, height)
        roi, x1p, y1p, x2p, y2p = make_padded_roi(frame, (x1, y1, x2, y2), PAD, width, height)

        if roi.size != 0:
            mp_res = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if mp_res.multi_hand_landmarks:
                # 가장 첫 손만 사용
                lms = mp_res.multi_hand_landmarks[0]
                for tip in fingertips:
                    cx = int(lms.landmark[tip].x * roi.shape[1] + x1p)
                    cy = int(lms.landmark[tip].y * roi.shape[0] + y1p)
                    mp_tip_points[tip] = np.array([cx, cy], dtype=np.float32)
                mp_ok = True

    # 4) MP 성공이면: tracked_tips 갱신 + miss_count 리셋
    if mp_ok:
        miss_count = 0
        for tip, pt in mp_tip_points.items():
            tracked_tips[tip] = pt.copy()
            if tip not in smoothed_tips:
                smoothed_tips[tip] = pt.copy()
            else:
                smoothed_tips[tip] = ema_vec(smoothed_tips[tip], pt, TIP_EMA_ALPHA)

    # 5) MP 실패면: Optical Flow로 fingertip 위치 예측 (짧게 버티기)
    else:
        miss_count += 1
        if prev_gray is not None and len(tracked_tips) > 0 and miss_count <= FLOW_MAX_MISS:
            # 이전 tips를 optical flow로 추적
            prev_pts = np.array([tracked_tips[t] for t in fingertips if t in tracked_tips], dtype=np.float32).reshape(-1, 1, 2)
            tip_ids = [t for t in fingertips if t in tracked_tips]

            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)

            if next_pts is not None and st is not None:
                for i, tip in enumerate(tip_ids):
                    if st[i, 0] == 1:
                        nx, ny = next_pts[i, 0]
                        # 화면 밖 튀는 거 방지
                        nx = float(np.clip(nx, 0, width - 1))
                        ny = float(np.clip(ny, 0, height - 1))
                        tracked_tips[tip] = np.array([nx, ny], dtype=np.float32)

                        if tip not in smoothed_tips:
                            smoothed_tips[tip] = tracked_tips[tip].copy()
                        else:
                            smoothed_tips[tip] = ema_vec(smoothed_tips[tip], tracked_tips[tip], TIP_EMA_ALPHA)
        else:
            # 너무 오래 못 찾으면 추적 리셋(틀린 점 계속 찍는 것 방지)
            if miss_count > FLOW_MAX_MISS:
                tracked_tips.clear()
                smoothed_tips.clear()

    # 6) smoothed_tips로 기존 로직(스트링/프렛 판정) 그대로 수행 + 시각화
    if prev_top_line is not None and len(smoothed_tips) > 0:
        for tip in fingertips:
            if tip not in smoothed_tips:
                continue

            cx, cy = smoothed_tips[tip]
            cx_i, cy_i = int(cx), int(cy)

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
                cv2.rectangle(display_frame, (cx_i+10, cy_i-th-5), (cx_i+10+tw, cy_i+5), (0, 0, 0), -1)
                cv2.putText(display_frame, txt, (cx_i+10, cy_i), 0, 0.5, (0, 255, 255), 2)
                cv2.circle(display_frame, (cx_i, cy_i), 6, (0, 0, 255), -1)

    prev_gray = gray

    # =========================
    # ✅ 화면 잘림 방지 + X로 닫으면 종료
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
