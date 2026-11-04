# run_tab_solver.py
from __future__ import annotations

import json
from pathlib import Path

from audio_vision_adapter import (
    load_json,
    load_audio_notes_from_audio_result,
    load_onsets_from_audio_result,
    build_vision_events_from_audio_notes,
)
from bass_fingering_solver import (
    solve_notes_greedy,
    merge_same_position_notes,
    solved_notes_to_dicts,
    export_ascii_tab,
)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_solve_tab(
    audio_result_json: str,
    vision_frames_json: str,
    out_dir: str = "outputs/solve",
    left_window: float = 0.10,
    right_window: float = 0.06,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_notes = load_audio_notes_from_audio_result(audio_result_json)
    vision_frames = load_json(vision_frames_json)

    vision_events = build_vision_events_from_audio_notes(
        vision_frames=vision_frames,
        audio_notes=audio_notes,
        left_window=left_window,
        right_window=right_window,
        min_frames=1,
    )

    solved = solve_notes_greedy(
        audio_notes=audio_notes,
        vision_events=vision_events,
        time_gate=max(left_window, right_window) + 0.05,
    )
    solved = merge_same_position_notes(solved, onsets=load_onsets_from_audio_result(audio_result_json))

    solved_dicts = solved_notes_to_dicts(solved)

    solved_json_path = out_dir / "solved_notes.json"
    events_json_path = out_dir / "vision_events.json"
    tab_txt_path = out_dir / "tab.txt"

    save_json(vision_events, str(events_json_path))
    save_json(solved_dicts, str(solved_json_path))
    export_ascii_tab(solved, str(tab_txt_path))

    return {
        "vision_events_json": str(events_json_path),
        "solved_notes_json": str(solved_json_path),
        "tab_txt": str(tab_txt_path),
        "solved_note_count": len(solved_dicts),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_result_json", required=True)
    ap.add_argument("--vision_frames_json", required=True)
    ap.add_argument("--out_dir", default="outputs/solve")
    ap.add_argument("--left_window", type=float, default=0.10)
    ap.add_argument("--right_window", type=float, default=0.06)

    args = ap.parse_args()

    result = run_solve_tab(
        audio_result_json=args.audio_result_json,
        vision_frames_json=args.vision_frames_json,
        out_dir=args.out_dir,
        left_window=args.left_window,
        right_window=args.right_window,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
