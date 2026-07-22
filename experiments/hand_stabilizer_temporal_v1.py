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
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])

def normalize(v: Tuple[float, float], eps: float = 1e-9) -> Tuple[float, float]:
    n = norm(v)
    if n < eps:
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)

def add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])

def sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])

def mul(v: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (v[0] * s, v[1] * s)

def clip_len(delta: Tuple[float, float], max_len: float) -> Tuple[float, float]:
    d = norm(delta)
    if d <= max_len or d < 1e-9:
        return delta
    return mul(delta, max_len / d)

def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t

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


# =========================
# MediaPipe landmark indices (thumb ignored)
# =========================
FINGERS = {
    "index":  {"dip": 7,  "tip": 8},
    "middle": {"dip": 11, "tip": 12},
    "ring":   {"dip": 15, "tip": 16},
    "pinky":  {"dip": 19, "tip": 20},
}

# for hand scale inside filter (index MCP=5, pinky MCP=17)
IDX_MCP = 5
PNK_MCP = 17


# =========================
# Filter (Combo1)
# =========================
@dataclass
class FingerState:
    tip_out: Optional[Tuple[float, float]] = None
    dip_out: Optional[Tuple[float, float]] = None
    tip_out_prev2: Optional[Tuple[float, float]] = None
    dip_out_prev2: Optional[Tuple[float, float]] = None
    L_ref: Optional[float] = None
    bad_count: int = 0

class TemporalAnatomicalHandFilter:
    def __init__(
        self,
        fps: float = 30.0,
        k_step: float = 0.25,
        w_len: float = 0.50,
        w_vel: float = 0.25,
        w_chain: float = 0.25,
        s_trust: float = 0.40,
        s_recon: float = 0.70,
        alpha_high: float = 0.75,
        alpha_low: float = 0.15,
        len_free: float = 0.15,
        len_full: float = 0.40,
        vel_free: float = 0.25,
        vel_full: float = 0.60,
        chain_free: float = 2.0,
        chain_full: float = 5.0,
        eps: float = 1e-6,
    ):
        self.fps = fps
        self.k_step = k_step
        self.w_len = w_len
        self.w_vel = w_vel
        self.w_chain = w_chain
        self.s_trust = s_trust
        self.s_recon = s_recon
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low
        self.len_free = len_free
        self.len_full = len_full
        self.vel_free = vel_free
        self.vel_full = vel_full
        self.chain_free = chain_free
        self.chain_full = chain_full
        self.eps = eps
        self.state: Dict[str, FingerState] = {k: FingerState() for k in FINGERS.keys()}

    def reset(self):
        self.state = {k: FingerState() for k in FINGERS.keys()}

    def update(self, lm_norm_full_21: List[Tuple[float, float]], W: int, H: int) -> Dict:
        lm_px = [(x * W, y * H) for (x, y) in lm_norm_full_21]

        S = dist(lm_px[IDX_MCP], lm_px[PNK_MCP])
        if S < 1.0:
            S = 1.0
        max_step = self.k_step * S

        out_px = list(lm_px)
        scores, modes = {}, {}

        for fname, idx in FINGERS.items():
            st = self.state[fname]
            dip_i, tip_i = idx["dip"], idx["tip"]
            DIP_obs, TIP_obs = lm_px[dip_i], lm_px[tip_i]

            if st.tip_out is None or st.dip_out is None:
                st.tip_out = TIP_obs
                st.dip_out = DIP_obs
                st.tip_out_prev2 = TIP_obs
                st.dip_out_prev2 = DIP_obs
                st.L_ref = max(dist(DIP_obs, TIP_obs), 1.0)
                scores[fname] = 0.0
                modes[fname] = "init"
                out_px[dip_i] = st.dip_out
                out_px[tip_i] = st.tip_out
                continue

            TIP_prev, DIP_prev = st.tip_out, st.dip_out

            TIP_clamp = add(TIP_prev, clip_len(sub(TIP_obs, TIP_prev), max_step))

            s_len = self._score_len(DIP_obs, TIP_clamp, st.L_ref)
            s_vel = self._score_vel(TIP_clamp, TIP_prev, S)
            s_chain = self._score_chain(TIP_clamp, TIP_prev, DIP_obs, DIP_prev)
            s = clamp01(self.w_len * s_len + self.w_vel * s_vel + self.w_chain * s_chain)

            mode = "normal"
            DIP_use = DIP_obs
            TIP_candidate = TIP_clamp

            if s >= self.s_recon:
                mode = "reconstruct"
                DIP_use = add(DIP_prev, clip_len(sub(DIP_obs, DIP_prev), 0.15 * S))

                dir_prev = normalize(sub(TIP_prev, DIP_prev), self.eps)
                Lref = st.L_ref if st.L_ref is not None else max(dist(DIP_use, TIP_prev), 1.0)
                TIP_candidate = add(DIP_use, mul(dir_prev, Lref))

            elif s >= self.s_trust:
                mode = "cautious"
                if s_len > 0.5 and st.L_ref is not None:
                    dir_prev = normalize(sub(TIP_prev, DIP_prev), self.eps)
                    TIP_candidate = add(DIP_obs, mul(dir_prev, st.L_ref))

            alpha = lerp(self.alpha_high, self.alpha_low, s)
            TIP_out = add(mul(TIP_candidate, alpha), mul(TIP_prev, 1.0 - alpha))

            dip_alpha = min(0.35, alpha)
            DIP_out = add(mul(DIP_use, dip_alpha), mul(DIP_prev, 1.0 - dip_alpha))

            if s < self.s_trust:
                L = dist(DIP_obs, TIP_out)
                st.L_ref = (0.90 * st.L_ref + 0.10 * max(L, 1.0)) if st.L_ref is not None else max(L, 1.0)

            st.tip_out_prev2 = st.tip_out
            st.dip_out_prev2 = st.dip_out
            st.tip_out = TIP_out
            st.dip_out = DIP_out

            out_px[dip_i] = DIP_out
            out_px[tip_i] = TIP_out
            scores[fname] = s
            modes[fname] = mode

        out_norm = [(x / W, y / H) for (x, y) in out_px]
        return {"landmarks_out": out_norm, "scores": scores, "modes": modes}

    def _score_len(self, DIP, TIP, L_ref):
        if L_ref is None or L_ref < 1e-3:
            return 0.0
        L = dist(DIP, TIP)
        eL = abs(L - L_ref) / (L_ref + self.eps)
        return clamp01((eL - self.len_free) / max(self.len_full - self.len_free, 1e-6))

    def _score_vel(self, TIP, TIP_prev, S):
        vN = dist(TIP, TIP_prev) / (S + self.eps)
        return clamp01((vN - self.vel_free) / max(self.vel_full - self.vel_free, 1e-6))

    def _score_chain(self, TIP, TIP_prev, DIP, DIP_prev):
        v_tip = dist(TIP, TIP_prev)
        v_dip = dist(DIP, DIP_prev)
        r = v_tip / (v_dip + self.eps)
        return clamp01((r - self.chain_free) / max(self.chain_full - self.chain_free, 1e-6))


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
    for fname, idx in FINGERS.items():
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
    parser.add_argument("--yolo_every", type=int, default=2, help="Run YOLO every N frames (speed). 1=every frame")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold for left_hand")
    parser.add_argument("--roi_margin", type=float, default=1.35, help="ROI margin scale around YOLO box")
    parser.add_argument("--mp_model_complexity", type=int, default=1, help="MediaPipe Hands model_complexity (0 faster, 1 better)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    # ---- PROJECT RELATIVE WEIGHT PATHS ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path_1 = os.path.join(base_dir, "weights", "weights.pt")
    weights_path_2 = os.path.join(base_dir, "weights", "weights_hands.pt")

    if not os.path.exists(weights_path_1):
        raise FileNotFoundError(weights_path_1)
    if not os.path.exists(weights_path_2):
        raise FileNotFoundError(weights_path_2)

    # Load YOLO models (CPU)
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

    window_name = "YOLO(left_hand)->ROI->MediaPipe + Combo1 (TIP only)"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_W, disp_H)

    # Writer (original resolution)
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # MediaPipe Hands on ROI
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=args.mp_model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    filt = TemporalAnatomicalHandFilter(fps=30.0)

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    last_box = None  # (x1,y1,x2,y2,conf)
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        overlay = frame.copy()

        # YOLO every N frames (with fallback between two weights)
        if (frame_idx % max(1, args.yolo_every)) == 0 or last_box is None:
            # 1) try model 1
            yres1 = yolo1.predict(frame, verbose=False, device="cpu")[0]
            best_left = select_left_hand_box_from_ultralytics_result(yres1, conf_thr=args.yolo_conf)

            # 2) fallback to model 2 if not found
            if best_left is None:
                yres2 = yolo2.predict(frame, verbose=False, device="cpu")[0]
                best_left = select_left_hand_box_from_ultralytics_result(yres2, conf_thr=args.yolo_conf)

            if best_left is not None:
                last_box = best_left

        if last_box is not None:
            x1, y1, x2, y2, conf = last_box
            rx1, ry1, rx2, ry2 = expand_box(x1, y1, x2, y2, W, H, margin=args.roi_margin)
            roi = frame[ry1:ry2, rx1:rx2]
            roi_h, roi_w = roi.shape[:2]

            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_roi)

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0].landmark
                lm_roi_norm = [(lm.x, lm.y) for lm in hand_lms]

                # Map ROI normalized -> FULL normalized
                lm_full_norm = []
                for (x, y) in lm_roi_norm:
                    gx = (rx1 + x * roi_w) / W
                    gy = (ry1 + y * roi_h) / H
                    lm_full_norm.append((gx, gy))

                # Filter on full frame
                out = filt.update(lm_full_norm, W=W, H=H)
                lm_f = out["landmarks_out"]

                # Draw ONLY filtered tips
                draw_filtered_tips_only(overlay, lm_f, color=(0, 255, 0), radius=4)

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
