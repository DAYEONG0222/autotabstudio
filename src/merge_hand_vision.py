# merge_vision_frames.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_left_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    left_frames.json frame ->
      {
        frame_idx, time,
        fret_candidates,
        left_string_candidates(optional debug)
      }
    """
    return {
        "frame_idx": int(frame.get("frame_idx", -1)),
        "time": float(frame["time"]),
        "fret_candidates": frame.get("left_fret_candidates", []),
        "left_string_candidates": frame.get("left_string_candidates", []),
    }


def normalize_right_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    right_frames.json frame ->
      {
        frame_idx, time,
        string_candidates
      }
    """
    return {
        "frame_idx": int(frame.get("frame_idx", -1)),
        "time": float(frame["time"]),
        "string_candidates": frame.get("right_string_candidates", []),
    }


def find_nearest_frame(
    target_time: float,
    frames: List[Dict[str, Any]],
    max_time_diff: float = 0.05
) -> Optional[Dict[str, Any]]:
    """
    Find nearest frame in time within max_time_diff.
    """
    best = None
    best_dt = float("inf")

    for fr in frames:
        dt = abs(float(fr["time"]) - target_time)
        if dt < best_dt and dt <= max_time_diff:
            best = fr
            best_dt = dt

    return best


def merge_left_right_frames(
    left_frames: List[Dict[str, Any]],
    right_frames: List[Dict[str, Any]],
    max_time_diff: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Merge by nearest time.
    Base timeline = union of left and right times.
    """
    left_norm = [normalize_left_frame(fr) for fr in left_frames]
    right_norm = [normalize_right_frame(fr) for fr in right_frames]

    all_times = sorted(set([float(fr["time"]) for fr in left_norm] + [float(fr["time"]) for fr in right_norm]))

    merged = []

    for i, t in enumerate(all_times):
        left_fr = find_nearest_frame(t, left_norm, max_time_diff=max_time_diff)
        right_fr = find_nearest_frame(t, right_norm, max_time_diff=max_time_diff)

        merged.append({
            "frame_idx": i,
            "time": float(t),
            "string_candidates": right_fr["string_candidates"] if right_fr else [],
            "fret_candidates": left_fr["fret_candidates"] if left_fr else [],
            "debug_left_string_candidates": left_fr["left_string_candidates"] if left_fr else [],
            "source_left_time": None if left_fr is None else float(left_fr["time"]),
            "source_right_time": None if right_fr is None else float(right_fr["time"]),
        })

    return merged


def main(left_json: str, right_json: str, out_json: str, max_time_diff: float = 0.05):
    left_frames = load_json(left_json)
    right_frames = load_json(right_json)

    merged = merge_left_right_frames(
        left_frames=left_frames,
        right_frames=right_frames,
        max_time_diff=max_time_diff
    )
    save_json(merged, out_json)

    print(f"Saved merged vision frames: {out_json}")
    print(f"Merged frame count: {len(merged)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--left_json", default="outputs/vision/left_frames.json")
    ap.add_argument("--right_json", default="outputs/vision/right_frames.json")
    ap.add_argument("--out_json", default="outputs/vision/vision_frames.json")
    ap.add_argument("--max_time_diff", type=float, default=0.05)

    args = ap.parse_args()

    main(
        left_json=args.left_json,
        right_json=args.right_json,
        out_json=args.out_json,
        max_time_diff=args.max_time_diff,
    )
