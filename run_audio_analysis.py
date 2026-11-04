from pathlib import Path

from src.audio.pyin_pitch_analyzer import run_pyin_analysis
from src.audio.audio_note_segmenter import detect_notes, save_analysis_result
from src.audio.onset_detector import detect_onsets
from src.audio_demucs_engine import AutotabAudioEngine, extract_bpm, extract_bpm_from_stems


def _resolve_analysis_audio_paths(
    session_audio_path: Path | None,
    source_path: Path,
    full_audio_path: Path,
    use_demucs: bool,
    demucs_output_root: Path,
) -> tuple[Path, Path | None]:
    if session_audio_path is not None:
        if not session_audio_path.exists():
            raise FileNotFoundError(f"session_audio_path not found: {session_audio_path}")
        return session_audio_path, None

    if not use_demucs:
        raise ValueError("session_audio_path is required unless --use_demucs is enabled.")

    demucs_input = full_audio_path if full_audio_path.exists() else source_path
    if not demucs_input.exists():
        raise FileNotFoundError(f"Demucs input not found: {demucs_input}")

    engine = AutotabAudioEngine(str(demucs_input), output_root=str(demucs_output_root))
    stems = engine.run_demucs()

    bass_path = Path(stems["bass"])
    drums_path = Path(stems["drums"])

    if not bass_path.exists():
        raise FileNotFoundError(f"Demucs bass stem not found: {bass_path}")
    if not drums_path.exists():
        raise FileNotFoundError(f"Demucs drums stem not found: {drums_path}")

    return bass_path, drums_path


def analyze_audio(
    session_audio_path: Path | None,
    output_json_path: Path,
    bpm: int | None,
    source_path: Path,
    full_audio_path: Path,
    step_size_ms: int = 10,
    fmin_hz: float = 40.0,
    fmax_hz: float = 300.0,
    max_midi_diff: float = 0.5,
    min_duration_sec: float = 0.1,
    use_demucs: bool = False,
    demucs_output_root: Path = Path("outputs/audio"),
    bpm_mode: str = "bass",
) -> None:
    """
    Analyze audio and save a JSON result compatible with src/video_adapter.py.
    Pitch is analyzed from bass audio. BPM can be analyzed from bass, drums,
    or a combined bass+drums stem tracker when Demucs stems are available.
    """
    print("\n" + "=" * 60)
    print("Audio analysis started")
    print("=" * 60)

    bass_audio_path, drums_audio_path = _resolve_analysis_audio_paths(
        session_audio_path=session_audio_path,
        source_path=source_path,
        full_audio_path=full_audio_path,
        use_demucs=use_demucs,
        demucs_output_root=demucs_output_root,
    )

    bpm_audio_path = drums_audio_path or bass_audio_path

    print(f"Bass audio: {bass_audio_path}")
    if drums_audio_path is not None:
        print(f"Drums audio: {drums_audio_path}")
    print(f"Output: {output_json_path}")
    print(f"BPM mode: {bpm_mode}")

    if bpm is None:
        if bpm_mode == "combined" and drums_audio_path is not None:
            estimated_bpm, _ = extract_bpm_from_stems(str(bass_audio_path), str(drums_audio_path))
            bpm = int(round(estimated_bpm))
            print(
                "BPM: auto-detected from bass+drums "
                f"{estimated_bpm:.2f} -> using {bpm}"
            )
            bpm_source_label = f"{bass_audio_path.name}+{drums_audio_path.name}"
        else:
            if bpm_mode == "bass":
                bpm_audio_path = bass_audio_path
            else:
                bpm_audio_path = drums_audio_path or bass_audio_path

            estimated_bpm, _ = extract_bpm(str(bpm_audio_path))
            bpm = int(round(estimated_bpm))
            print(f"BPM: auto-detected from {bpm_audio_path.name} {estimated_bpm:.2f} -> using {bpm}")
            bpm_source_label = bpm_audio_path.name
    else:
        print(f"BPM: {bpm}")
        bpm_source_label = "manual"

    sr, times, freqs, midi, conf = run_pyin_analysis(
        audio_path=bass_audio_path,
        step_size_ms=step_size_ms,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )

    notes = detect_notes(
        times=times,
        frequencies=freqs,
        midi_notes=midi,
        confidences=conf,
        max_midi_diff=max_midi_diff,
        min_duration_sec=min_duration_sec,
    )
    onsets = detect_onsets(bass_audio_path)

    video_exts = {".mp4", ".mov", ".mkv", ".avi"}
    source_type = "video" if source_path.suffix.lower() in video_exts else "audio"

    metadata = {
        "bpm": int(bpm),
        "source_type": source_type,
        "source_path": source_path,
        "full_audio_path": full_audio_path,
        "session_audio_path": bass_audio_path,
        "sample_rate": int(sr),
        "pitch_backend": "librosa-pyin",
        "step_size_ms": int(step_size_ms),
        "fmin_hz": float(fmin_hz),
        "fmax_hz": float(fmax_hz),
        "bpm_source_path": bpm_audio_path,
        "bpm_mode": bpm_mode,
        "bpm_source_label": bpm_source_label,
    }

    save_analysis_result(
        output_path=output_json_path,
        notes=notes,
        metadata=metadata,
        onsets=onsets,
    )

    print("\n" + "=" * 60)
    print("Audio analysis finished")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--session_audio_path", default=None)
    ap.add_argument("--output_json_path", default="outputs/audio/result.json")
    ap.add_argument("--bpm", type=int, default=None)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--full_audio_path", required=True)
    ap.add_argument("--step_size_ms", type=int, default=10)
    ap.add_argument("--fmin_hz", type=float, default=40.0)
    ap.add_argument("--fmax_hz", type=float, default=300.0)
    ap.add_argument("--max_midi_diff", type=float, default=0.5)
    ap.add_argument("--min_duration_sec", type=float, default=0.1)
    ap.add_argument("--use_demucs", action="store_true")
    ap.add_argument("--demucs_output_root", default="outputs/audio")
    ap.add_argument("--bpm_mode", choices=["combined", "drums", "bass"], default="bass")

    args = ap.parse_args()

    analyze_audio(
        session_audio_path=None if args.session_audio_path is None else Path(args.session_audio_path),
        output_json_path=Path(args.output_json_path),
        bpm=args.bpm,
        source_path=Path(args.source_path),
        full_audio_path=Path(args.full_audio_path),
        step_size_ms=args.step_size_ms,
        fmin_hz=args.fmin_hz,
        fmax_hz=args.fmax_hz,
        max_midi_diff=args.max_midi_diff,
        min_duration_sec=args.min_duration_sec,
        use_demucs=args.use_demucs,
        demucs_output_root=Path(args.demucs_output_root),
        bpm_mode=args.bpm_mode,
    )
