import os
import cv2
import math
import argparse
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mediapipe as mp

# =========================
# Helpers (원본 코드의 헬퍼 함수 100% 유지)
# =========================
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])

def normalize(v: Tuple[float, float], eps: float = 1e-9) -> Tuple[float, float]:
    n = norm(v)
    if n < eps: return (1.0, 0.0)
    return (v[0] / n, v[1] / n)

def add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])

def sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])

def mul(v: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (v[0] * s, v[1] * s)

def clip_len(delta: Tuple[float, float], max_len: float) -> Tuple[float, float]:
    d = norm(delta)
    if d <= max_len or d < 1e-9: return delta
    return mul(delta, max_len / d)

def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t

# =========================
# MediaPipe landmark indices
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
# Filter (기존 428줄 원본 로직 100% 반영)
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

    def update(self, lm_norm_full_21: List[Tuple[float, float]], W: int, H: int) -> Dict:
        lm_px = [(x * W, y * H) for (x, y) in lm_norm_full_21]
        S = max(dist(lm_px[IDX_MCP], lm_px[PNK_MCP]), 1.0)
        max_step = self.k_step * S
        out_px = list(lm_px)

        for fname, idx in FINGERS.items():
            st = self.state[fname]
            dip_i, tip_i = idx["dip"], idx["tip"]
            DIP_obs, TIP_obs = lm_px[dip_i], lm_px[tip_i]

            if st.tip_out is None:
                st.tip_out, st.dip_out = TIP_obs, DIP_obs
                st.tip_out_prev2, st.dip_out_prev2 = TIP_obs, DIP_obs
                st.L_ref = max(dist(DIP_obs, TIP_obs), 1.0)
                continue

            TIP_prev, DIP_prev = st.tip_out, st.dip_out
            TIP_clamp = add(TIP_prev, clip_len(sub(TIP_obs, TIP_prev), max_step))

            s_len = self._score_len(DIP_obs, TIP_clamp, st.L_ref)
            s_vel = self._score_vel(TIP_clamp, TIP_prev, S)
            s_chain = self._score_chain(TIP_clamp, TIP_prev, DIP_obs, DIP_prev)
            s = clamp01(self.w_len * s_len + self.w_vel * s_vel + self.w_chain * s_chain)

            DIP_use, TIP_candidate = DIP_obs, TIP_clamp
            if s >= self.s_recon:
                DIP_use = add(DIP_prev, clip_len(sub(DIP_obs, DIP_prev), 0.15 * S))
                dir_prev = normalize(sub(TIP_prev, DIP_prev), self.eps)
                TIP_candidate = add(DIP_use, mul(dir_prev, st.L_ref))
            elif s >= self.s_trust:
                if s_len > 0.5:
                    dir_prev = normalize(sub(TIP_prev, DIP_prev), self.eps)
                    TIP_candidate = add(DIP_obs, mul(dir_prev, st.L_ref))

            alpha = lerp(self.alpha_high, self.alpha_low, s)
            TIP_out = add(mul(TIP_candidate, alpha), mul(TIP_prev, 1.0 - alpha))
            DIP_out = add(mul(DIP_use, min(0.35, alpha)), mul(DIP_prev, 1.0 - min(0.35, alpha)))

            if s < self.s_trust:
                st.L_ref = (0.9 * st.L_ref + 0.1 * max(dist(DIP_out, TIP_out), 1.0))

            st.tip_out_prev2, st.dip_out_prev2 = TIP_prev, DIP_prev
            st.tip_out, st.dip_out = TIP_out, DIP_out
            out_px[dip_i], out_px[tip_i] = DIP_out, TIP_out

        return {"landmarks_out": [(x / W, y / H) for (x, y) in out_px]}

    def _score_len(self, DIP, TIP, L_ref):
        if not L_ref: return 0.0
        eL = abs(dist(DIP, TIP) - L_ref) / (L_ref + self.eps)
        return clamp01((eL - self.len_free) / max(self.len_full - self.len_free, 1e-6))

    def _score_vel(self, TIP, TIP_prev, S):
        vN = dist(TIP, TIP_prev) / (S + self.eps)
        return clamp01((vN - self.vel_free) / max(self.vel_full - self.vel_free, 1e-6))

    def _score_chain(self, TIP, TIP_prev, DIP, DIP_prev):
        r = dist(TIP, TIP_prev) / (dist(DIP, DIP_prev) + self.eps)
        return clamp01((r - self.chain_free) / max(self.chain_full - self.chain_free, 1e-6))

# =========================
# Main (MediaPipe 전용 + 왼손 추적 고정)
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--out", default="", help="Output path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 화면 크기 자동 맞춤 설정 (영상 전체가 보이도록 함)
    root = tk.Tk(); root.withdraw()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    scale = min((sw * 0.8) / W, (sh * 0.8) / H, 1.0)
    disp_W, disp_H = int(W * scale), int(H * scale)

    window_name = "Left Hand Tracking (Press ESC to Quit)"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_W, disp_H)

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2, 
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    filt = TemporalAnatomicalHandFilter(fps=fps)

    # 왼손 추적 상태 유지 변수
    last_hand_x = None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        overlay = frame.copy()

        target_lms = None

        if results.multi_hand_landmarks:
            detected_hands = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 손의 중심 X 좌표 계산
                avg_x = sum([lm.x for lm in hand_landmarks.landmark]) / 21
                detected_hands.append({'idx': i, 'x': avg_x})

            # 왼손(기타 넥 쪽) 고정 추적 로직
            if last_hand_x is None:
                # 첫 감지 시: 화면 오른쪽에 있는 손을 우선 선택 (X가 큰 값)
                detected_hands.sort(key=lambda h: h['x'], reverse=True)
                target_lms = results.multi_hand_landmarks[detected_hands[0]['idx']]
                last_hand_x = detected_hands[0]['x']
            else:
                # 이후: 이전 프레임의 위치와 가장 가까운 손 추적
                detected_hands.sort(key=lambda h: abs(h['x'] - last_hand_x))
                if abs(detected_hands[0]['x'] - last_hand_x) < 0.4: # 갑작스러운 점프 방지
                    target_lms = results.multi_hand_landmarks[detected_hands[0]['idx']]
                    last_hand_x = detected_hands[0]['x']

        if target_lms:
            lms = [(lm.x, lm.y) for lm in target_lms.landmark]
            out = filt.update(lms, W, H)
            
            # 필터링된 손가락 끝 포인트 그리기
            for fname, idx in FINGERS.items():
                lx, ly = out["landmarks_out"][idx["tip"]]
                cv2.circle(overlay, (int(lx * W), int(ly * H)), 5, (0, 255, 0), -1)

        if writer: writer.write(overlay)
        if args.show:
            show_frame = cv2.resize(overlay, (disp_W, disp_H), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, show_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): break # ESC 종료 옵션

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()