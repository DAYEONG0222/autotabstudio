from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEFAULT_VIDEO = Path("data/videos/test_video6.mp4")
DEFAULT_WEIGHTS = Path("data/weights/weights.pt")
DEFAULT_OUTPUT_JSON = Path("outputs/vision/left_frames.json")

cv2 = None
mp = None
np = None
torch = None
YOLO = None


def load_vision_dependencies():
    global YOLO, cv2, mp, np, torch

    if cv2 is not None:
        return

    import cv2 as cv2_module
    import mediapipe as mp_module
    import numpy as np_module
    import torch as torch_module
    from ultralytics import YOLO as yolo_class

    cv2 = cv2_module
    mp = mp_module
    np = np_module
    torch = torch_module
    YOLO = yolo_class


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        a_d = (2 * np.pi * self.d_cutoff * t_e) / (2 * np.pi * self.d_cutoff * t_e + 1)
        dx_hat = a_d * ((x - self.x_prev) / t_e) + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * np.pi * cutoff * t_e) / (2 * np.pi * cutoff * t_e + 1)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.t_prev = t
        self.dx_prev = dx_hat
        return x_hat


class PointFilter:
    def __init__(self, t0, x0, y0, min_cutoff=1.0, beta=0.005):
        self.x_filter = OneEuroFilter(t0, x0, min_cutoff=min_cutoff, beta=beta)
        self.y_filter = OneEuroFilter(t0, y0, min_cutoff=min_cutoff, beta=beta)

    def process(self, t, pt):
        return self.x_filter.filter(t, pt[0]), self.y_filter.filter(t, pt[1])


def get_y_on_line(params, x):
    if params is None or x is None:
        return 0
    vx, vy, x0, y0 = params
    return y0 + ((x - x0) / (vx if vx != 0 else 1e-6)) * vy


def smart_sort_and_label_v6(pts, nut_center):
    dists = [np.linalg.norm(pt - nut_center) for pt in pts]
    sorted_indices = np.argsort(dists)
    near_pair = pts[sorted_indices[:2]]
    far_pair = pts[sorted_indices[2:]]
    near_pair = near_pair[np.argsort(near_pair[:, 1])]
    far_pair = far_pair[np.argsort(far_pair[:, 1])]
    poly_pts = np.array([near_pair[0], far_pair[0], far_pair[1], near_pair[1]], dtype="float32")
    axis_vec = np.mean(pts, axis=0) - nut_center
    is_reversed = np.cross(axis_vec, poly_pts[0] - nut_center) > 0
    if is_reversed:
        tops, bots = [poly_pts[3], poly_pts[2]], [poly_pts[0], poly_pts[1]]
    else:
        tops, bots = [poly_pts[0], poly_pts[1]], [poly_pts[3], poly_pts[2]]
    return poly_pts, tops, bots, np.max(dists)


def refine_per_finger(mp_lms, roi_offset, roi_shape, nut_center, hand_mask):
    if not mp_lms or nut_center is None:
        return {}

    fingertips = [8, 12, 16, 20]
    raw_points = []
    for idx in fingertips:
        lm = mp_lms.landmark[idx]
        tx = int(lm.x * roi_shape[1] + roi_offset[0])
        ty = int(lm.y * roi_shape[0] + roi_offset[1])
        raw_points.append(np.array([tx, ty]))

    raw_points.sort(key=lambda p: np.linalg.norm(p - nut_center))

    result = {}
    for i, idx in enumerate(fingertips):
        pt = raw_points[i]
        in_mask = False
        if 0 <= pt[1] < hand_mask.shape[0] and 0 <= pt[0] < hand_mask.shape[1]:
            in_mask = bool(hand_mask[pt[1], pt[0]] > 0)
        result[idx] = {"pt": pt, "in_mask": in_mask}
    return result


def run_left_hand_fret_detection(
    video_path: str | Path = DEFAULT_VIDEO,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    display: bool = True,
) -> list[dict]:
    load_vision_dependencies()

    video_path = Path(video_path)
    weights_path = Path(weights_path)
    output_json = Path(output_json)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not found. Running on CPU.")

    seg_model = YOLO(str(weights_path)).to(device)
    hands_mp = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    string_ratios = [k / 8.0 for k in [0.9, 2.85, 4.85, 6.8]]

    finger_filters = {}
    prev_t, prev_b, px_min, px_max, last_nut = None, None, None, None, None
    left_frame_results = []
    frame_idx = 0
    window_name = "Autotab Left Hand Fret Detection"

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_t = time.time()
        display_frame = frame.copy()
        frame_time = frame_idx / fps
        frame_record = {
            "frame_idx": int(frame_idx),
            "time": float(frame_time),
            "left_string_candidates": [],
            "left_fret_candidates": [],
        }

        results = seg_model.predict(frame, conf=0.3, verbose=False, device=device)[0]
        hand_mask = np.zeros((height, width), dtype=np.uint8)
        hand_detected, hand_box, fret_data = False, None, []

        if results.masks:
            for i, contour in enumerate(results.masks.xy):
                cls = seg_model.names[int(results.boxes.cls[i])]
                if cls == "nut":
                    last_nut = np.mean(contour, axis=0)
                elif cls == "left_hand":
                    cv2.fillPoly(hand_mask, [contour.astype(np.int32)], 255)
                    hand_detected = True
                    hand_box = results.boxes.xyxy[i].cpu().numpy()
                elif cls == "fret_zone" and last_nut is not None:
                    poly, tops, bots, dist = smart_sort_and_label_v6(
                        cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour.astype(np.int32)))),
                        last_nut,
                    )
                    fret_data.append({"poly": poly, "tops": tops, "bots": bots, "dist": dist})

        fret_data.sort(key=lambda x: x["dist"])
        all_top, all_bot, all_x = [], [], []
        if len(fret_data) >= 3:
            for fret_obj in fret_data:
                all_top.extend(fret_obj["tops"])
                all_bot.extend(fret_obj["bots"])
                all_x.extend([pt[0] for pt in fret_obj["poly"]])
                if display:
                    cv2.polylines(display_frame, [fret_obj["poly"].astype(np.int32)], True, (255, 255, 100), 2)

            if len(all_top) >= 4:
                prev_t = cv2.fitLine(np.array(all_top, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
                prev_b = cv2.fitLine(np.array(all_bot, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
                px_min, px_max = min(all_x) - 50, max(all_x) + 50

        if display and prev_t is not None and prev_b is not None and px_min is not None and px_max is not None:
            for ratio in string_ratios:
                p1 = (
                    int(px_min),
                    int(get_y_on_line(prev_t, px_min) * (1 - ratio) + get_y_on_line(prev_b, px_min) * ratio),
                )
                p2 = (
                    int(px_max),
                    int(get_y_on_line(prev_t, px_max) * (1 - ratio) + get_y_on_line(prev_b, px_max) * ratio),
                )
                cv2.line(display_frame, p1, p2, (50, 255, 50), 1)

        final_results = {}
        if hand_detected and last_nut is not None:
            x1, y1, x2, y2 = map(int, hand_box)
            pad = 30
            roi_rect = (max(0, x1 - pad), max(0, y1 - pad))
            roi = frame[roi_rect[1] : min(height, y2 + pad), roi_rect[0] : min(width, x2 + pad)]

            if roi.size > 0:
                mp_res = hands_mp.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                if mp_res.multi_hand_landmarks:
                    finger_data = refine_per_finger(mp_res.multi_hand_landmarks[0], roi_rect, roi.shape, last_nut, hand_mask)
                    for idx, data in finger_data.items():
                        pt, in_mask = data["pt"], data["in_mask"]
                        if idx not in finger_filters:
                            finger_filters[idx] = PointFilter(curr_t, pt[0], pt[1])
                        fx, fy = finger_filters[idx].process(curr_t, pt)
                        final_results[idx] = {"coord": (int(fx), int(fy)), "in_mask": in_mask}

        for idx, res in final_results.items():
            cx, cy = res["coord"]
            in_mask = res["in_mask"]

            if display:
                cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)

            if prev_t is None or prev_b is None:
                continue

            mdist, bass_string = float("inf"), -1
            for s_idx, ratio in enumerate(string_ratios):
                ty = get_y_on_line(prev_t, cx) * (1 - ratio) + get_y_on_line(prev_b, cx) * ratio
                if abs(cy - ty) < mdist:
                    mdist, bass_string = abs(cy - ty), 4 - s_idx

            current_fret = 0
            for fret_num, fret_obj in enumerate(fret_data):
                if cv2.pointPolygonTest(fret_obj["poly"], (float(cx), float(cy)), False) >= 0:
                    current_fret = fret_num + 1
                    break

            if bass_string != -1:
                string_conf = max(0.0, 1.0 - (mdist / 80.0))
                frame_record["left_string_candidates"].append({
                    "string": int(bass_string),
                    "conf": float(string_conf),
                    "finger_idx": int(idx),
                    "x": int(cx),
                    "y": int(cy),
                    "in_mask": bool(in_mask),
                    "mdist": float(mdist),
                })
                frame_record["left_fret_candidates"].append({
                    "fret": int(current_fret),
                    "conf": 1.0 if current_fret > 0 else 0.2,
                    "finger_idx": int(idx),
                    "x": int(cx),
                    "y": int(cy),
                    "in_mask": bool(in_mask),
                })

            if display and mdist < 45:
                label = f"S{bass_string} F{current_fret}" + ("*" if in_mask else "")
                cv2.putText(display_frame, label, (cx + 10, cy), 0, 0.5, (0, 0, 255), 2)

        left_frame_results.append(frame_record)
        frame_idx += 1

        if display:
            cv2.putText(display_frame, f"Frame: {frame_idx} Time: {frame_time:.3f}s ESC: Exit", (20, 35), 0, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(left_frame_results, f, ensure_ascii=False, indent=2)

    cap.release()
    hands_mp.close()
    if display:
        cv2.destroyWindow(window_name)

    print(f"Saved left-hand vision frames: {output_json}")
    print(f"Total frames: {len(left_frame_results)}")
    return left_frame_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--no_display", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_left_hand_fret_detection(
        video_path=args.video,
        weights_path=args.weights,
        output_json=args.output_json,
        display=not args.no_display,
    )
