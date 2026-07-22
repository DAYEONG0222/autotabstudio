# audio_vision_adapter.py
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_frame_candidates(frame: Dict[str, Any]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Supports both:
      1) single-value frame:
         {
           "time": 1.0, "string": 2, "fret": 5,
           "string_conf": 0.8, "fret_conf": 0.9
         }

      2) candidate-list frame:
         {
           "time": 1.0,
           "string_candidates": [{"string":2,"conf":0.8}, ...],
           "fret_candidates": [{"fret":5,"conf":0.9}, ...]
         }
    """
    if "string_candidates" in frame:
        string_cands = [
            {"string": int(x["string"]), "conf": float(x.get("conf", 1.0))}
            for x in frame["string_candidates"]
        ]
    elif "string" in frame and frame["string"] is not None:
        string_cands = [{
            "string": int(frame["string"]),
            "conf": float(frame.get("string_conf", frame.get("conf", 1.0)))
        }]
    else:
        string_cands = []

    if "fret_candidates" in frame:
        fret_cands = [
            {"fret": int(x["fret"]), "conf": float(x.get("conf", 1.0))}
            for x in frame["fret_candidates"]
        ]
    elif "fret" in frame and frame["fret"] is not None:
        fret_cands = [{
            "fret": int(frame["fret"]),
            "conf": float(frame.get("fret_conf", frame.get("conf", 1.0)))
        }]
    else:
        fret_cands = []

    return string_cands, fret_cands


def collect_frames_in_window(
    vision_frames: List[Dict[str, Any]],
    center_t: float,
    left_window: float,
    right_window: float
) -> List[Dict[str, Any]]:
    t0 = center_t - left_window
    t1 = center_t + right_window
    return [fr for fr in vision_frames if t0 <= float(fr["time"]) <= t1]


def aggregate_candidates(frames: List[Dict[str, Any]]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Aggregate per-frame candidates into event-level candidates.
    Strategy:
      - sum confidences for same string/fret
      - normalize to probability-like scores
    """
    string_scores = defaultdict(float)
    fret_scores = defaultdict(float)

    for fr in frames:
        string_cands, fret_cands = normalize_frame_candidates(fr)

        for c in string_cands:
            string_scores[int(c["string"])] += float(c.get("conf", 1.0))

        for c in fret_cands:
            fret_scores[int(c["fret"])] += float(c.get("conf", 1.0))

    def normalize_score_map(score_map, key_name):
        total = sum(score_map.values())
        if total <= 0:
            return []
        items = [
            {key_name: int(k), "conf": float(v / total)}
            for k, v in sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        ]
        return items

    return normalize_score_map(string_scores, "string"), normalize_score_map(fret_scores, "fret")


def build_vision_events_from_audio_notes(
    vision_frames: List[Dict[str, Any]],
    audio_notes: List[Dict[str, Any]],
    left_window: float = 0.10,
    right_window: float = 0.06,
    min_frames: int = 1
) -> List[Dict[str, Any]]:
    """
    Build event-level vision candidates centered on each audio note onset.
    Output format:
      [
        {
          "t": audio_note_t_on,
          "string_candidates": [...],
          "fret_candidates": [...],
          "matched_frame_count": N
        },
        ...
      ]
    """
    events = []

    for note in audio_notes:
        t_on = float(note["t_on"])
        frames = collect_frames_in_window(
            vision_frames=vision_frames,
            center_t=t_on,
            left_window=left_window,
            right_window=right_window
        )

        if len(frames) < min_frames:
            events.append({
                "t": t_on,
                "string_candidates": [],
                "fret_candidates": [],
                "matched_frame_count": 0
            })
            continue

        string_cands, fret_cands = aggregate_candidates(frames)

        events.append({
            "t": t_on,
            "string_candidates": string_cands,
            "fret_candidates": fret_cands,
            "matched_frame_count": len(frames)
        })

    return events


def load_audio_notes_from_audio_result(audio_result_json: str) -> List[Dict[str, Any]]:
    """
    Converts run_audio_analysis.py result.json -> solver input notes.
    Expects result.json to contain midi_notes list with start/end/pitch.
    """
    data = load_json(audio_result_json)
    midi_notes = data.get("midi_notes") or []

    notes = []
    for n in midi_notes:
        notes.append({
            "t_on": float(n["start"]),
            "t_off": float(n["end"]),
            "pitch": int(n["pitch"]),
            "conf": float(n.get("conf", 1.0)),
        })

    notes.sort(key=lambda x: x["t_on"])
    return notes
