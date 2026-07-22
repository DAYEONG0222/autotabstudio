from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def run_pyin_analysis(
    audio_path: Path,
    step_size_ms: int = 10,
    fmin_hz: float = 40.0,
    fmax_hz: float = 300.0,
    frame_length: int = 2048,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract frame-level pitch information with librosa.pyin.
    """
    audio, sr = sf.read(str(audio_path))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    hop_length = max(1, int(sr * step_size_ms / 1000))

    print("[Pitch analysis] started")
    print(f"  file: {audio_path.name}")
    print(f"  sample_rate: {sr}")
    print(f"  step_size_ms: {step_size_ms}")
    print(f"  fmin_hz: {fmin_hz}")
    print(f"  fmax_hz: {fmax_hz}")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    frequencies = np.array(f0, dtype=float)
    confidences = np.array(voiced_prob, dtype=float)

    invalid_mask = np.isnan(frequencies)
    confidences[invalid_mask] = 0.0

    midi_notes = np.full_like(frequencies, np.nan, dtype=float)
    valid_mask = ~invalid_mask & (frequencies > 0)
    midi_notes[valid_mask] = 69 + 12 * np.log2(frequencies[valid_mask] / 440.0)

    total_frames = len(times)
    valid_frames = int(np.sum(valid_mask))
    print("[Pitch analysis] finished")
    print(f"  total_frames: {total_frames}")
    print(f"  valid_frames: {valid_frames}")

    return sr, times, frequencies, midi_notes, confidences


def analyze_pitch_statistics(
    frequencies: np.ndarray,
    midi_notes: np.ndarray,
    confidences: np.ndarray,
) -> dict:
    valid_mask = ~np.isnan(frequencies)

    stats = {
        "valid_count": int(np.sum(valid_mask)),
        "total_count": len(frequencies),
        "valid_ratio": float(np.mean(valid_mask)) if len(frequencies) else 0.0,
    }

    if stats["valid_count"] > 0:
        valid_freqs = frequencies[valid_mask]
        valid_midi = midi_notes[valid_mask]
        valid_conf = confidences[valid_mask]
        stats.update(
            {
                "freq_min": float(np.min(valid_freqs)),
                "freq_max": float(np.max(valid_freqs)),
                "freq_mean": float(np.mean(valid_freqs)),
                "midi_min": float(np.min(valid_midi)),
                "midi_max": float(np.max(valid_midi)),
                "midi_mean": float(np.mean(valid_midi)),
                "confidence_mean": float(np.mean(valid_conf)),
            }
        )
    else:
        stats.update(
            {
                "freq_min": None,
                "freq_max": None,
                "freq_mean": None,
                "midi_min": None,
                "midi_max": None,
                "midi_mean": None,
                "confidence_mean": 0.0,
            }
        )

    return stats
