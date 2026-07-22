from ultralytics import YOLO
import cv2
import numpy as np
import time
import csv
from pathlib import Path
import ctypes

# ============================================================
# 설정 (프로젝트 루트 기준 상대경로)
# ============================================================
BASS_MODEL_PATH = r'./weights_bass_left_hand.pt'   # 넥( nut/fret_zone )용
HAND_MODEL_PATH = r'./weights_hand_only.pt'        # 손 키포인트(4개 fingertip)용
VIDEO_PATH      = r'./bassimages/test_video.mp4'

OUT_VIDEO_PATH  = r'output_hand_coords.mp4'
OUT_CSV_PATH    = r'hand_coords.csv'

# 워핑(정규화 넥 ROI) 크기
WARP_W, WARP_H = 900, 240

# Optical Flow 파라미터
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# YOLO 파라미터
CONF_BASS = 0.25
IOU_BASS  = 0.5

# 손은 작고 가려져서 conf 낮추는 게 유리
CONF_HAND = 0.10
IOU_HAND  = 0.5

# 트래킹 드리프트 방지: N프레임마다 넥 모델로 리프레시(0이면 끔)
REFRESH_EVERY = 30

# 네 손 키포인트 라벨(학습한 순서대로 맞춰 쓰면 됨)
# Roboflow에서 찍은 순서가 "검지, 중지, 약지, 소지" 라고 했으니 이렇게 가정
FINGER_NAMES = ["index", "middle", "ring", "pinky"]


# ============================================================
# 화면에 맞춰 자동 축소 + 창 리사이즈 유틸
# ============================================================
def get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def resize_to_fit(img, max_w, max_h):
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img, 1.0
    scale = min(max_w / w, max_h / h)
    if scale >= 1.0:
        return img, 1.0
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


# ============================================================
# 넥 코너 정렬 (너 코드 기반)
# ============================================================
def smart_sort_corners(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair = [pts[sorted_indices[0]], pts[sorted_indices[1]]]
    far_pair  = [pts[sorted_indices[2]], pts[sorted_indices[3]]]

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

def get_point_on_fitline(vx, vy, x0, y0, x):
    if abs(vx) < 1e-3:
        return np.array([x0, y0], dtype=np.float32)
    t = (x - x0) / vx
    y = y0 + t * vy
    return np.array([x, y], dtype=np.float32)


# ============================================================
# Homography Warp
# ============================================================
def warp_neck(frame_bgr, p_st, p_et, p_eb, p_sb):
    src = np.array([p_st, p_et, p_eb, p_sb], dtype=np.float32)
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame_bgr, H, (WARP_W, WARP_H))
    return warped, H

def transform_point(H, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


# ============================================================
# [핵심] 손 키포인트(4개 fingertip) 추출
# - result.keypoints.xy : (num_det, K, 2)
# - result.keypoints.conf : (num_det, K)  (버전에 따라 없을 수 있음)
# 반환:
#   best_pt: (x, y, conf, kpt_idx)
#   all_pts: [(x,y,conf,kpt_idx), ...]  (K개)
# ============================================================
def extract_fingertips_from_result(result, min_kpt_conf=0.15):
    if getattr(result, "keypoints", None) is None or result.keypoints is None:
        return None, []

    try:
        xy = result.keypoints.xy  # tensor
    except Exception:
        return None, []

    if xy is None or len(xy) == 0:
        return None, []

    # detection 여러 개면 "평균 conf"가 가장 높은 detection 선택
    conf_mat = getattr(result.keypoints, "conf", None)

    det_idx = 0
    if conf_mat is not None:
        try:
            det_scores = conf_mat.mean(dim=1).cpu().numpy()
            det_idx = int(np.argmax(det_scores))
        except Exception:
            det_idx = 0

    kpts = xy[det_idx].cpu().numpy()  # (K,2)
    if conf_mat is not None:
        kconf = conf_mat[det_idx].cpu().numpy()  # (K,)
    else:
        kconf = np.ones((kpts.shape[0],), dtype=np.float32)

    all_pts = []
    best = None
    for i in range(kpts.shape[0]):
        x, y = float(kpts[i, 0]), float(kpts[i, 1])
        c = float(kconf[i]) if i < len(kconf) else 1.0
        if c < min_kpt_conf:
            continue
        all_pts.append((x, y, c, i))
        if best is None or c > best[2]:
            best = (x, y, c, i)

    return best, all_pts


# ============================================================
# 넥 앵커 스캔: nut + fret_zone로 dual regression anchor 만들기
# ============================================================
def scan_neck_anchors(model_bass, frame_bgr):
    results = model_bass.predict(frame_bgr, conf=CONF_BASS, iou=IOU_BASS, verbose=False)
    result = results[0]

    nut_center = None
    all_top_points = []
    all_bot_points = []
    all_x_coords = []

    if result.masks is None:
        return None

    contours = result.masks.xy

    # nut 찾기
    for i, contour in enumerate(contours):
        if len(contour) > 0 and model_bass.names[int(result.boxes.cls[i])] == 'nut':
            box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.int32)))
            nut_center = np.mean(box, axis=0)

    if nut_center is None:
        return None

    # fret_zone 수집
    for i, contour in enumerate(contours):
        if len(contour) > 0 and model_bass.names[int(result.boxes.cls[i])] == 'fret_zone':
            cnt = contour.astype(np.int32)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            sorted_pts = smart_sort_corners(box, nut_center)
            all_top_points.extend([sorted_pts[0], sorted_pts[1]])
            all_bot_points.extend([sorted_pts[2], sorted_pts[3]])
            all_x_coords.extend([pt[0] for pt in sorted_pts])

    if len(all_top_points) < 6:
        return None

    pts_top = np.array(all_top_points, dtype=np.float32)
    line_top = cv2.fitLine(pts_top, cv2.DIST_L2, 0, 0.01, 0.01)
    vx_t, vy_t, x0_t, y0_t = [float(v) for v in line_top]

    pts_bot = np.array(all_bot_points, dtype=np.float32)
    line_bot = cv2.fitLine(pts_bot, cv2.DIST_L2, 0, 0.01, 0.01)
    vx_b, vy_b, x0_b, y0_b = [float(v) for v in line_bot]

    x_min = float(min(all_x_coords))
    x_max = float(max(all_x_coords))

    p_st = get_point_on_fitline(vx_t, vy_t, x0_t, y0_t, x_min)
    p_sb = get_point_on_fitline(vx_b, vy_b, x0_b, y0_b, x_min)
    p_et = get_point_on_fitline(vx_t, vy_t, x0_t, y0_t, x_max)
    p_eb = get_point_on_fitline(vx_b, vy_b, x0_b, y0_b, x_max)

    return p_st, p_sb, p_et, p_eb


# ============================================================
# main
# ============================================================
def main():
    for p in [BASS_MODEL_PATH, HAND_MODEL_PATH, VIDEO_PATH]:
        if not Path(p).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {p}")
            print("   - 프로젝트 루트에서 실행 중인지 확인하세요.")
            return

    model_bass = YOLO(BASS_MODEL_PATH)
    model_hand = YOLO(HAND_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, float(fps_in), (w, h))

    csv_file = open(OUT_CSV_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_idx", "t_sec", "neck_locked",
                     "best_x", "best_y", "best_conf", "best_kpt",
                     "kpt0_x","kpt0_y","kpt0_conf",
                     "kpt1_x","kpt1_y","kpt1_conf",
                     "kpt2_x","kpt2_y","kpt2_conf",
                     "kpt3_x","kpt3_y","kpt3_conf"
                     ])

    # 창 1개만!
    cv2.namedWindow("Hand Coords (Original)", cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_size()
    MARGIN_W, MARGIN_H = 80, 120
    max_w = max(320, screen_w - MARGIN_W)
    max_h = max(240, screen_h - MARGIN_H)

    is_tracking = False
    p0 = None
    old_gray = None
    frame_idx = 0
    last_refresh = 0

    print("▶ 시작: 넥 Auto-Lock → 워핑에서 손 키포인트(4개) 추정 → 원본에 좌표 표시/저장")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        # --------------------------------------------------------
        # 1) 넥 앵커 업데이트
        # --------------------------------------------------------
        if is_tracking and p0 is not None and old_gray is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is None or st is None or st.reshape(-1).sum() < 4:
                is_tracking = False
                p0 = None
                old_gray = None
            else:
                p_st, p_sb, p_et, p_eb = p1.reshape(4, 2)
                old_gray = frame_gray.copy()
                p0 = p1.reshape(-1, 1, 2)

                # 리프레시로 드리프트 보정
                if REFRESH_EVERY > 0 and (frame_idx - last_refresh) >= REFRESH_EVERY:
                    anchors = scan_neck_anchors(model_bass, frame)
                    last_refresh = frame_idx
                    if anchors is not None:
                        p_st2, p_sb2, p_et2, p_eb2 = anchors
                        d = np.mean(np.linalg.norm(
                            np.array([p_st2, p_sb2, p_et2, p_eb2]) - np.array([p_st, p_sb, p_et, p_eb]),
                            axis=1
                        ))
                        if d < 60:
                            p_st, p_sb, p_et, p_eb = p_st2, p_sb2, p_et2, p_eb2
                            p0 = np.array([p_st, p_sb, p_et, p_eb], dtype=np.float32).reshape(-1, 1, 2)
                            old_gray = frame_gray.copy()

        if not is_tracking:
            anchors = scan_neck_anchors(model_bass, frame)
            if anchors is not None:
                p_st, p_sb, p_et, p_eb = anchors
                p0 = np.array([p_st, p_sb, p_et, p_eb], dtype=np.float32).reshape(-1, 1, 2)
                old_gray = frame_gray.copy()
                is_tracking = True

        # --------------------------------------------------------
        # 2) 워핑 + 손 키포인트 추정
        # --------------------------------------------------------
        best_pt_orig = None
        all_pts_orig = []  # 최대 4개 기대

        if is_tracking:
            # 넥 4등분 라인
            for k in range(1, 4):
                r = k / 4.0
                p_start = p_st * (1 - r) + p_sb * r
                p_end   = p_et * (1 - r) + p_eb * r
                cv2.line(disp, tuple(p_start.astype(int)), tuple(p_end.astype(int)), (0, 255, 128), 2)

            warped, H = warp_neck(frame, p_st, p_et, p_eb, p_sb)
            Hinv = np.linalg.inv(H)

            # 손 키포인트는 워핑 ROI에서 추정
            res_hand = model_hand.predict(warped, conf=CONF_HAND, iou=IOU_HAND, verbose=False)[0]

            best_w, all_w = extract_fingertips_from_result(res_hand, min_kpt_conf=0.10)

            # 디버그 텍스트: det 수, kpt 수
            det_count = len(res_hand.boxes) if res_hand.boxes is not None else 0
            kpt_count = 0
            if getattr(res_hand, "keypoints", None) is not None and res_hand.keypoints is not None:
                try:
                    kpt_count = int(res_hand.keypoints.xy.shape[1])
                except Exception:
                    kpt_count = 0

            cv2.putText(disp, f"MODE: TRACKING + WARP + HAND-KPT", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)
            cv2.putText(disp, f"hand det:{det_count}  kpts:{kpt_count}  conf:{CONF_HAND}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 모든 fingertip(4개)을 원본 좌표로 변환해서 찍기
            for (wx, wy, c, ki) in all_w:
                ox, oy = transform_point(Hinv, wx, wy)
                all_pts_orig.append((ox, oy, c, ki))

            # best도 원본으로
            if best_w is not None:
                wx, wy, c, ki = best_w
                ox, oy = transform_point(Hinv, wx, wy)
                best_pt_orig = (ox, oy, c, ki)

            # 원본 프레임에 표시 (예시처럼 점 + 라벨)
            for (ox, oy, c, ki) in all_pts_orig:
                name = FINGER_NAMES[ki] if ki < len(FINGER_NAMES) else f"kpt{ki}"
                cv2.circle(disp, (int(ox), int(oy)), 10, (0, 255, 255), -1)
                cv2.putText(disp, f"{name} {c:.2f}", (int(ox) + 8, int(oy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # best는 더 크게 표시
            if best_pt_orig is not None:
                ox, oy, c, ki = best_pt_orig
                cv2.circle(disp, (int(ox), int(oy)), 14, (0, 0, 255), 3)

        else:
            cv2.putText(disp, "MODE: SCANNING (finding neck)...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --------------------------------------------------------
        # 3) CSV 기록 (best + 4개 kpt)
        # --------------------------------------------------------
        row = [frame_idx, f"{t_sec:.3f}", int(is_tracking)]

        if best_pt_orig is None:
            row += ["", "", "", ""]
        else:
            bx, by, bc, bk = best_pt_orig
            row += [f"{bx:.2f}", f"{by:.2f}", f"{bc:.3f}", int(bk)]

        # 4개 키포인트 기록 (없으면 빈칸)
        # ki=0..3에 해당하는 값으로 정렬해서 넣기
        kpt_map = {ki: (x, y, c) for (x, y, c, ki) in all_pts_orig}
        for ki in range(4):
            if ki in kpt_map:
                x, y, c = kpt_map[ki]
                row += [f"{x:.2f}", f"{y:.2f}", f"{c:.3f}"]
            else:
                row += ["", "", ""]

        writer.writerow(row)

        # --------------------------------------------------------
        # 4) 저장 + 화면 표시 (화면에 맞춰 자동 축소)
        # --------------------------------------------------------
        out.write(disp)
        disp_show, _ = resize_to_fit(disp, max_w, max_h)
        cv2.imshow("Hand Coords (Original)", disp_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    csv_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 완료: {OUT_VIDEO_PATH}, {OUT_CSV_PATH} 저장됨")


if __name__ == "__main__":
    main()