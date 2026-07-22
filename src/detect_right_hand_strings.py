from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEFAULT_VIDEO = Path("data/videos/test_video6.mp4")
DEFAULT_SEG_WEIGHTS = Path("data/weights/weights.pt")
DEFAULT_HAND_WEIGHTS = Path("data/weights/weights_hands.pt")
DEFAULT_OUTPUT_JSON = Path("outputs/vision/right_frames.json")

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


def run_right_hand_string_detection(
    video_path: str | Path = DEFAULT_VIDEO,
    seg_weights_path: str | Path = DEFAULT_SEG_WEIGHTS,
    hand_weights_path: str | Path = DEFAULT_HAND_WEIGHTS,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    display: bool = True,
) -> list[dict]:
    load_vision_dependencies()

    video_path = Path(video_path)
    seg_weights_path = Path(seg_weights_path)
    hand_weights_path = Path(hand_weights_path)
    output_json = Path(output_json)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not seg_weights_path.exists():
        raise FileNotFoundError(f"Segmentation weights not found: {seg_weights_path}")
    if not hand_weights_path.exists():
        raise FileNotFoundError(f"Hand weights not found: {hand_weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = YOLO(str(seg_weights_path)).to(device)
    hand_model = YOLO(str(hand_weights_path)).to(device)
    hands_mp = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
    finger_history = {4: [], 8: [], 12: [], 16: []}
    history_len = 6
    move_threshold = 3.5
    candidate_threshold = 35
    prev_t, prev_b, last_nut = None, None, None
    right_frame_results = []
    frame_idx = 0
    window_name = "Autotab Right Hand String Detection"

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        display_frame = frame.copy()
        frame_time = frame_idx / fps
        frame_record = {
            "frame_idx": int(frame_idx),
            "time": float(frame_time),
            "right_string_candidates": [],
            "right_hand_detected": False,
            "right_hand_box": None,
            "finger_points": [],
        }

        res_seg = seg_model.predict(frame, conf=0.3, verbose=False, device=device)[0]
        res_hand = hand_model.predict(frame, conf=0.3, verbose=False, device=device)[0]
        fret_data, right_hand_box = [], None

        if res_seg.masks:
            for i, contour in enumerate(res_seg.masks.xy):
                cls = seg_model.names[int(res_seg.boxes.cls[i])]
                if cls == "nut":
                    last_nut = np.mean(contour, axis=0)
                elif cls == "fret_zone" and last_nut is not None:
                    poly, tops, bots, dist = smart_sort_and_label_v6(
                        cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour.astype(np.int32)))),
                        last_nut,
                    )
                    fret_data.append({"poly": poly, "tops": tops, "bots": bots, "dist": dist})

        if len(res_hand.boxes) > 0:
            for box in res_hand.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                if ((bx1 + bx2) / 2) < (width / 4):
                    right_hand_box = [bx1, by1, bx2, by2]
                    break

        if right_hand_box is not None:
            frame_record["right_hand_detected"] = True
            frame_record["right_hand_box"] = [float(v) for v in right_hand_box]

        if len(fret_data) >= 3:
            fret_data.sort(key=lambda x: x["dist"])
            all_top, all_bot = [], []
            for fret_obj in fret_data:
                all_top.extend(fret_obj["tops"])
                all_bot.extend(fret_obj["bots"])

            if len(all_top) >= 4:
                prev_t = cv2.fitLine(np.array(all_top, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
                prev_b = cv2.fitLine(np.array(all_bot, dtype=np.int32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

        candidate_strings = set()
        candidate_scores = {}

        if right_hand_box is not None and prev_t is not None and prev_b is not None:
            rx1, ry1, rx2, ry2 = map(int, right_hand_box)
            roi_rect = (max(0, rx1 - 20), max(0, ry1 - 20))
            roi_img = frame[roi_rect[1] : min(height, ry2 + 20), roi_rect[0] : min(width, rx2 + 20)]

            if roi_img.size > 0:
                mp_res = hands_mp.process(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
                if mp_res.multi_hand_landmarks:
                    for hand_lms in mp_res.multi_hand_landmarks:
                        for idx in [4, 8, 12, 16]:
                            lm = hand_lms.landmark[idx]
                            raw_cy = int(lm.y * roi_img.shape[0] + roi_rect[1])
                            cx = int(lm.x * roi_img.shape[1] + roi_rect[0])

                            if idx not in finger_filters:
                                finger_filters[idx] = OneEuroFilter(curr_time, raw_cy, min_cutoff=1.0, beta=0.01)
                            filtered_cy = finger_filters[idx].filter(curr_time, raw_cy)

                            finger_history[idx].append(filtered_cy)
                            if len(finger_history[idx]) > history_len:
                                finger_history[idx].pop(0)

                            movement = float(np.std(finger_history[idx])) if len(finger_history[idx]) >= 3 else 0.0
                            is_active = movement > move_threshold
                            nearest_string, nearest_dist = -1, float("inf")

                            for s_idx, ratio in enumerate(string_ratios):
                                line_y = get_y_on_line(prev_t, cx) * (1 - ratio) + get_y_on_line(prev_b, cx) * ratio
                                dist = abs(filtered_cy - line_y)
                                if dist < nearest_dist:
                                    nearest_dist = dist
                                    nearest_string = 4 - s_idx

                                if dist < candidate_threshold and is_active:
                                    cand = 4 - s_idx
                                    candidate_strings.add(cand)
                                    conf = max(0.0, 1.0 - (dist / max(candidate_threshold, 1)))
                                    candidate_scores[cand] = max(candidate_scores.get(cand, 0.0), conf)

                            frame_record["finger_points"].append({
                                "finger_idx": int(idx),
                                "x": int(cx),
                                "y_raw": int(raw_cy),
                                "y_filtered": float(filtered_cy),
                                "movement": float(movement),
                                "is_active": bool(is_active),
                                "nearest_string": int(nearest_string) if nearest_string != -1 else None,
                                "distance_to_string": float(nearest_dist) if nearest_dist != float("inf") else None,
                            })

                            if display:
                                point_color = (255, 0, 255) if is_active else (150, 150, 150)
                                cv2.circle(display_frame, (cx, int(filtered_cy)), 6, point_color, -1)

            if display:
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                cv2.putText(display_frame, f"Active Cand: {sorted(candidate_strings)}", (rx1, ry1 - 15), 0, 0.6, (0, 255, 255), 2)

        for string_num in sorted(candidate_strings):
            frame_record["right_string_candidates"].append({
                "string": int(string_num),
                "conf": float(candidate_scores.get(string_num, 1.0)),
            })

        right_frame_results.append(frame_record)
        frame_idx += 1

        if display:
            if prev_t is not None and prev_b is not None:
                for ratio in string_ratios:
                    p1 = (0, int(get_y_on_line(prev_t, 0) * (1 - ratio) + get_y_on_line(prev_b, 0) * ratio))
                    p2 = (width, int(get_y_on_line(prev_t, width) * (1 - ratio) + get_y_on_line(prev_b, width) * ratio))
                    cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_idx} Time: {frame_time:.3f}s ESC: Exit", (20, 35), 0, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(right_frame_results, f, ensure_ascii=False, indent=2)

    cap.release()
    hands_mp.close()
    if display:
        cv2.destroyWindow(window_name)

    print(f"Saved right-hand vision frames: {output_json}")
    print(f"Total frames: {len(right_frame_results)}")
    return right_frame_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--seg_weights", default=str(DEFAULT_SEG_WEIGHTS))
    parser.add_argument("--hand_weights", default=str(DEFAULT_HAND_WEIGHTS))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--no_display", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_right_hand_string_detection(
        video_path=args.video,
        seg_weights_path=args.seg_weights,
        hand_weights_path=args.hand_weights,
        output_json=args.output_json,
        display=not args.no_display,
    )
