from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np


def detect_notes(
    times: np.ndarray,
    frequencies: np.ndarray,
    midi_notes: np.ndarray,
    confidences: np.ndarray,
    max_midi_diff: float = 0.5,
    min_duration_sec: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Convert frame-level pitch results into note segments.
    """
    n_frames = len(times)
    note_segments = []
    current_start = None

    print("[Note detection] started")
    print(f"  frames: {n_frames}")
    print(f"  max_midi_diff: {max_midi_diff}")
    print(f"  min_duration_sec: {min_duration_sec}")

    for i in range(n_frames):
        if np.isnan(frequencies[i]):
            if current_start is not None:
                note_segments.append((current_start, i - 1))
                current_start = None
            continue

        if current_start is None:
            current_start = i
            continue

        if np.isnan(midi_notes[i]) or np.isnan(midi_notes[i - 1]):
            note_segments.append((current_start, i - 1))
            current_start = None
            continue

        if abs(midi_notes[i] - midi_notes[i - 1]) > max_midi_diff:
            note_segments.append((current_start, i - 1))
            current_start = i

    if current_start is not None:
        note_segments.append((current_start, n_frames - 1))

    notes: List[Dict[str, Any]] = []
    note_id = 0

    for start_idx, end_idx in note_segments:
        onset = float(times[start_idx])
        frame_end = float(times[end_idx])
        frame_step = float(times[1] - times[0]) if len(times) > 1 else 0.0
        offset = frame_end + frame_step
        duration = max(offset - onset, 0.0)

        if duration < min_duration_sec:
            continue

        freq_segment = frequencies[start_idx : end_idx + 1]
        midi_segment = midi_notes[start_idx : end_idx + 1]
        conf_segment = confidences[start_idx : end_idx + 1]

        freq_mean = float(np.nanmean(freq_segment))
        midi_mean = float(np.nanmean(midi_segment))
        conf_mean = 0.0 if np.all(np.isnan(conf_segment)) else float(np.nanmean(conf_segment))

        notes.append(
            {
                "id": note_id,
                "time_sec": onset,
                "duration_sec": duration,
                "end_sec": offset,
                "freq_hz": freq_mean,
                "midi": midi_mean,
                "pitch": int(round(midi_mean)),
                "confidence_mean": conf_mean,
            }
        )
        note_id += 1

    print("[Note detection] finished")
    print(f"  notes: {len(notes)}")
    return notes


def _build_midi_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    midi_notes = []
    for note in notes:
        start = float(note["time_sec"])
        end = float(note.get("end_sec", start + float(note["duration_sec"])))
        midi_notes.append(
            {
                "start": start,
                "end": end,
                "pitch": int(note["pitch"]),
                "conf": float(note.get("confidence_mean", 1.0)),
            }
        )
    return midi_notes


def save_analysis_result(
    output_path: Path,
    notes: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    onsets: List[float] | None = None,
) -> None:
    """
    Save analysis JSON. Includes both a rich 'notes' section and a
    'midi_notes' section compatible with src/video_adapter.py.
    'onsets' (attack times in seconds) is written when provided.
    """
    midi_notes = _build_midi_notes(notes)

    data = {
        "bpm": metadata["bpm"],
        "source": {
            "type": metadata["source_type"],
            "path": str(metadata["source_path"]),
        },
        "audio": {
            "full_path": str(metadata["full_audio_path"]),
            "session_path": str(metadata["session_audio_path"]),
            "sample_rate": metadata["sample_rate"],
        },
        "pitch": {
            "backend": metadata["pitch_backend"],
            "step_size_ms": metadata["step_size_ms"],
            "fmin_hz": metadata["fmin_hz"],
            "fmax_hz": metadata["fmax_hz"],
        },
        "notes": notes,
        "midi_notes": midi_notes,
    }
    if onsets is not None:
        data["onsets"] = onsets

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("[Result saved]")
    print(f"  file: {output_path}")
    print(f"  notes: {len(notes)}")
    print(f"  midi_notes: {len(midi_notes)}")


def load_analysis_result(input_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Analysis result not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("[Result loaded]")
    print(f"  file: {input_path}")
    print(f"  notes: {len(data.get('notes', []))}")
    print(f"  midi_notes: {len(data.get('midi_notes', []))}")
    return data
