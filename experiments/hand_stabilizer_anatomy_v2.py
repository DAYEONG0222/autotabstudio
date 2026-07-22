# autotab_yolo_left_hand_mediapipe_combo2_tiponly_v2.py
# Combo2 (Anatomy-first + EMA) with robustness improvements:
# 1) YOLO two-weights fallback
# 2) Larger ROI support (roi_margin)
# 3) MediaPipe fail -> retry once with bigger ROI
# 4) MediaPipe fail -> HOLD last good tips for a few frames (no blinking)
# 5) No boxes, ONLY filtered fingertip points are drawn
# 6) Display auto-fit to screen
#
# Requirements:
#   pip install ultralytics opencv-python mediapipe
#
# Run:
#   python autotab_yolo_left_hand_mediapipe_combo2_tiponly_v2.py --video test_video6.mp4 --show
#   python autotab_yolo_left_hand_mediapipe_combo2_tiponly_v2.py --video test_video6.mp4 --out out_combo2.mp4 --show
#
# Recommended knobs (CPU 30fps target):
#   --roi_margin 1.7 --yolo_every 1
#   If slow: --yolo_every 2

import os
import cv2
import math
import argparse
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
from ultralytics import YOLO


# =========================
# Helpers
# =========================
def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])

def add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])

def sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])

def mul(v: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (v[0] * s, v[1] * s)

def normalize(v: Tuple[float, float], eps: float = 1e-9) -> Tuple[float, float]:
    n = norm(v)
    if n < eps:
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1 + 1: x2 = min(W - 1, x1 + 2)
    if y2 <= y1 + 1: y2 = min(H - 1, y1 + 2)
    return x1, y1, x2, y2

def expand_box(x1, y1, x2, y2, W, H, margin: float = 1.35):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * margin
    bh = (y2 - y1) * margin
    nx1 = cx - bw / 2
    ny1 = cy - bh / 2
    nx2 = cx + bw / 2
    ny2 = cy + bh / 2
    return clip_box(nx1, ny1, nx2, ny2, W, H)

def angle_between(u: Tuple[float, float], v: Tuple[float, float], eps: float = 1e-9) -> float:
    nu, nv = norm(u), norm(v)
    if nu < eps or nv < eps:
        return 0.0
    dot = (u[0] * v[0] + u[1] * v[1]) / (nu * nv + eps)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


# =========================
# MediaPipe landmark indices (thumb ignored)
# =========================
FINGERS = {
    "index":  {"dip": 7,  "tip": 8},
    "middle": {"dip": 11, "tip": 12},
    "ring":   {"dip": 15, "tip": 16},
    "pinky":  {"dip": 19, "tip": 20},
}
IDX_MCP = 5
PNK_MCP = 17


# =========================
# Combo2 Filter: Anatomy-first + EMA
# =========================
@dataclass
class FingerState2:
    tip_out: Optional[Tuple[float, float]] = None
    dip_out: Optional[Tuple[float, float]] = None
    dir_prev: Optional[Tuple[float, float]] = None
    L_ref: Optional[float] = None


class AnatomyFirstHandFilterCombo2:
    """
    Combo2:
      - Always project TIP based on a stable DIP->TIP length (L_ref) and a stable direction.
      - Temporal component is EMA smoothing only (no scoring gate).
      - Robustness: if direction jumps too much vs previous, reuse previous direction.
    """
    def __init__(
        self,
        alpha_tip: float = 0.65,
        alpha_dip: float = 0.35,
        L_ref_beta: float = 0.05,
        max_dir_jump_deg: float = 70.0,
        min_len_px: float = 1.0,
        eps: float = 1e-6,
    ):
        self.alpha_tip = clamp01(alpha_tip)
        self.alpha_dip = clamp01(alpha_dip)
        self.L_ref_beta = clamp01(L_ref_beta)
        self.max_dir_jump = math.radians(max_dir_jump_deg)
        self.min_len_px = min_len_px
        self.eps = eps
        self.state: Dict[str, FingerState2] = {k: FingerState2() for k in FINGERS.keys()}

    def reset(self):
        self.state = {k: FingerState2() for k in FINGERS.keys()}

    def update(self, lm_norm_full_21: List[Tuple[float, float]], W: int, H: int) -> Dict:
        lm_px = [(x * W, y * H) for (x, y) in lm_norm_full_21]
        out_px = list(lm_px)

        # optional scale (not strictly needed here, but could be used later)
        _S = dist(lm_px[IDX_MCP], lm_px[PNK_MCP])
        if _S < 1.0:
            _S = 1.0

        for fname, idx in FINGERS.items():
            st = self.state[fname]
            dip_i, tip_i = idx["dip"], idx["tip"]
            DIP_obs = lm_px[dip_i]
            TIP_obs = lm_px[tip_i]

            if st.tip_out is None or st.dip_out is None:
                st.tip_out = TIP_obs
                st.dip_out = DIP_obs
                d_obs = sub(TIP_obs, DIP_obs)
                st.dir_prev = normalize(d_obs, self.eps)
                st.L_ref = max(dist(DIP_obs, TIP_obs), self.min_len_px)
                out_px[dip_i] = st.dip_out
                out_px[tip_i] = st.tip_out
                continue

            # Smooth DIP (helps projection stability)
            DIP_prev = st.dip_out
            DIP_out = add(mul(DIP_obs, self.alpha_dip), mul(DIP_prev, 1.0 - self.alpha_dip))

            # Direction decision
            d_obs = sub(TIP_obs, DIP_obs)
            dir_obs = normalize(d_obs, self.eps)

            dir_use = dir_obs
            if st.dir_prev is not None:
                ang = angle_between(dir_obs, st.dir_prev, self.eps)
                if ang > self.max_dir_jump:
                    dir_use = st.dir_prev

            # Update L_ref only when observed length is within a stable band
            L_obs = max(dist(DIP_obs, TIP_obs), self.min_len_px)
            if st.L_ref is None:
                st.L_ref = L_obs
            else:
                ratio = L_obs / (st.L_ref + self.eps)
                if 0.85 <= ratio <= 1.15:
                    st.L_ref = (1.0 - self.L_ref_beta) * st.L_ref + self.L_ref_beta * L_obs

            Lref = st.L_ref if st.L_ref is not None else L_obs

            # Anatomy-first projection
            TIP_proj = add(DIP_out, mul(dir_use, Lref))

            # Temporal smoothing on projected TIP
            TIP_prev = st.tip_out
            TIP_out = add(mul(TIP_proj, self.alpha_tip), mul(TIP_prev, 1.0 - self.alpha_tip))

            # Store
            st.dip_out = DIP_out
            st.tip_out = TIP_out
            st.dir_prev = normalize(sub(TIP_out, DIP_out), self.eps)

            out_px[dip_i] = DIP_out
            out_px[tip_i] = TIP_out

        out_norm = [(x / W, y / H) for (x, y) in out_px]
        return {"landmarks_out": out_norm}


# =========================
# YOLO left-hand selection
# =========================
def select_left_hand_box_from_ultralytics_result(yres, conf_thr: float) -> Optional[Tuple[int, int, int, int, float]]:
    boxes = yres.boxes
    if boxes is None or len(boxes) == 0:
        return None
    names = yres.names
    best = None
    for b in boxes:
        cls_id = int(b.cls.item())
        name = names.get(cls_id, str(cls_id))
        conf = float(b.conf.item())
        if name != "left_hand":
            continue
        if conf < conf_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        cand = (int(x1), int(y1), int(x2), int(y2), conf)
        if best is None or cand[4] > best[4]:
            best = cand
    return best


# =========================
# Draw ONLY fingertip points
# =========================
def draw_filtered_tips_only(frame, lm_norm_full, color=(0, 255, 0), radius=4):
    H, W = frame.shape[:2]
    for _, idx in FINGERS.items():
        tip_i = idx["tip"]
        x, y = lm_norm_full[tip_i]
        cv2.circle(frame, (int(x * W), int(y * H)), radius, color, -1)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input mp4 path")
    parser.add_argument("--out", default="", help="Optional output mp4 path (e.g., out.mp4)")
    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")

    # YOLO / ROI knobs
    parser.add_argument("--yolo_every", type=int, default=2, help="Run YOLO every N frames (speed). 1=every frame")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold for left_hand")
    parser.add_argument("--roi_margin", type=float, default=1.7, help="ROI margin scale around YOLO box")
    parser.add_argument("--retry_roi_margin", type=float, default=1.9, help="Retry ROI margin if MP fails")

    # MediaPipe knobs
    parser.add_argument("--mp_model_complexity", type=int, default=1, help="MediaPipe Hands model_complexity (0 faster, 1 better)")
    parser.add_argument("--mp_det", type=float, default=0.4, help="MediaPipe min_detection_confidence")
    parser.add_argument("--mp_trk", type=float, default=0.4, help="MediaPipe min_tracking_confidence")

    # Combo2 knobs
    parser.add_argument("--alpha_tip", type=float, default=0.65, help="Combo2 TIP EMA alpha (0..1)")
    parser.add_argument("--alpha_dip", type=float, default=0.35, help="Combo2 DIP EMA alpha (0..1)")
    parser.add_argument("--dir_jump_deg", type=float, default=70.0, help="Max direction jump deg before using prev direction")
    parser.add_argument("--lref_beta", type=float, default=0.05, help="L_ref update beta (small=stable)")

    # Hold-last knobs
    parser.add_argument("--hold_frames", type=int, default=8, help="Hold last good tips when MP misses (frames)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    # ---- FIXED WEIGHT PATHS (set your 2nd weights path!) ----
    weights_path_1 = r"C:\Users\82107\OneDrive\바탕 화면\多映\2026\AutoTabStudio\weights.pt"
    weights_path_2 = r"C:\Users\82107\OneDrive\바탕 화면\多映\2026\AutoTabStudio\weights_hands.pt"  # <-- CHANGE THIS

    if not os.path.exists(weights_path_1):
        raise FileNotFoundError(weights_path_1)
    if not os.path.exists(weights_path_2):
        raise FileNotFoundError(weights_path_2)

    # Load YOLO models
    yolo1 = YOLO(weights_path_1)
    yolo2 = YOLO(weights_path_2)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Display auto-fit
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    margin_w, margin_h = 80, 140
    fit_w = max(200, screen_w - margin_w)
    fit_h = max(200, screen_h - margin_h)
    display_scale = min(fit_w / W, fit_h / H, 1.0)
    disp_W, disp_H = int(W * display_scale), int(H * display_scale)

    window_name = "YOLO(left_hand)->ROI->MediaPipe + Combo2 (TIP only) v2"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_W, disp_H)

    # Writer (original resolution)
    writer = None
    if args.out:
        out_lower = args.out.lower()

        # Choose codec by extension for Windows-friendly playback
        if out_lower.endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        elif out_lower.endswith(".mp4"):
            # Try H264 first (may fail if not supported), fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        else:
            # default to avi-style codec if extension is unknown
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

        writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

        # If avc1 fails, fallback to mp4v for mp4
        if not writer.isOpened() and out_lower.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

        if not writer.isOpened():
            raise RuntimeError(
                f"VideoWriter failed to open. Try saving as .avi (XVID). Out={args.out}"
            )

    # MediaPipe Hands on ROI
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=args.mp_model_complexity,
        min_detection_confidence=float(args.mp_det),
        min_tracking_confidence=float(args.mp_trk),
    )

    filt = AnatomyFirstHandFilterCombo2(
        alpha_tip=clamp01(args.alpha_tip),
        alpha_dip=clamp01(args.alpha_dip),
        max_dir_jump_deg=float(args.dir_jump_deg),
        L_ref_beta=clamp01(args.lref_beta),
    )

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # State: YOLO box + hold-last
    last_box = None  # (x1,y1,x2,y2,conf)
    last_good_lm_f = None
    miss_count = 0
    HOLD_FRAMES = max(0, int(args.hold_frames))

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        overlay = frame.copy()

        # --- YOLO every N frames (fallback across two weights) ---
        if (frame_idx % max(1, int(args.yolo_every))) == 0 or last_box is None:
            yres1 = yolo1.predict(frame, verbose=False, device="cpu")[0]
            best_left = select_left_hand_box_from_ultralytics_result(yres1, conf_thr=float(args.yolo_conf))

            if best_left is None:
                yres2 = yolo2.predict(frame, verbose=False, device="cpu")[0]
                best_left = select_left_hand_box_from_ultralytics_result(yres2, conf_thr=float(args.yolo_conf))

            if best_left is not None:
                last_box = best_left

        drew_tips_this_frame = False

        if last_box is not None:
            x1, y1, x2, y2, _conf = last_box

            # Primary ROI
            rx1, ry1, rx2, ry2 = expand_box(x1, y1, x2, y2, W, H, margin=float(args.roi_margin))
            roi = frame[ry1:ry2, rx1:rx2]
            roi_h, roi_w = roi.shape[:2]

            # MediaPipe on primary ROI
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_roi)

            # Retry once with bigger ROI if MP fails
            if not results.multi_hand_landmarks:
                rx1b, ry1b, rx2b, ry2b = expand_box(
                    x1, y1, x2, y2, W, H,
                    margin=max(float(args.roi_margin), float(args.retry_roi_margin))
                )
                roi2 = frame[ry1b:ry2b, rx1b:rx2b]
                rgb_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)
                results2 = hands.process(rgb_roi2)

                if results2.multi_hand_landmarks:
                    results = results2
                    rx1, ry1, rx2, ry2 = rx1b, ry1b, rx2b, ry2b
                    roi_h, roi_w = roi2.shape[:2]

            if results.multi_hand_landmarks:
                miss_count = 0

                hand_lms = results.multi_hand_landmarks[0].landmark
                lm_roi_norm = [(lm.x, lm.y) for lm in hand_lms]

                # ROI normalized -> FULL normalized
                lm_full_norm = []
                for (x, y) in lm_roi_norm:
                    gx = (rx1 + x * roi_w) / W
                    gy = (ry1 + y * roi_h) / H
                    lm_full_norm.append((gx, gy))

                out = filt.update(lm_full_norm, W=W, H=H)
                lm_f = out["landmarks_out"]

                last_good_lm_f = lm_f
                draw_filtered_tips_only(overlay, lm_f, color=(0, 255, 0), radius=4)
                drew_tips_this_frame = True

            else:
                # MP miss: hold last good tips
                miss_count += 1
                if last_good_lm_f is not None and miss_count <= HOLD_FRAMES:
                    draw_filtered_tips_only(overlay, last_good_lm_f, color=(0, 255, 0), radius=4)
                    drew_tips_this_frame = True

        # If no box or no tips drawn and no hold, nothing is drawn (by design)

        if writer:
            writer.write(overlay)

        if args.show:
            show_frame = overlay
            if display_scale < 1.0:
                show_frame = cv2.resize(overlay, (disp_W, disp_H), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, show_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    hands.close()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    if args.out:
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
