from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str]) -> None:
    print("\n" + "=" * 72)
    print(" ".join(command))
    print("=" * 72)
    subprocess.run(command, check=True)


def run_full_pipeline(
    video: str,
    *,
    audio_json: str = "outputs/audio/result.json",
    left_json: str = "outputs/vision/left_frames.json",
    right_json: str = "outputs/vision/right_frames.json",
    vision_json: str = "outputs/vision/vision_frames.json",
    solve_dir: str = "outputs/solve",
    musicxml: str = "outputs/musicxml/tab_output.musicxml",
    no_display: bool = False,
    use_demucs: bool = True,
    bpm_mode: str = "combined",
    skip_audio: bool = False,
    skip_vision_detection: bool = False,
    skip_vision_merge: bool = False,
) -> None:
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    python = sys.executable

    if skip_audio:
        if not Path(audio_json).exists():
            raise FileNotFoundError(f"Audio JSON not found: {audio_json}")
    else:
        audio_cmd = [
            python,
            "run_audio_analysis.py",
            "--source_path",
            str(video_path),
            "--full_audio_path",
            str(video_path),
            "--output_json_path",
            audio_json,
            "--bpm_mode",
            bpm_mode,
        ]
        if use_demucs:
            audio_cmd.append("--use_demucs")
        else:
            audio_cmd.extend(["--session_audio_path", str(video_path)])
        run_step(audio_cmd)

    display_arg = ["--no_display"] if no_display else []
    if skip_vision_detection:
        if not Path(left_json).exists():
            raise FileNotFoundError(f"Left-hand vision JSON not found: {left_json}")
        if not Path(right_json).exists():
            raise FileNotFoundError(f"Right-hand vision JSON not found: {right_json}")
    else:
        run_step([
            python,
            "src/detect_left_hand_frets.py",
            "--video",
            str(video_path),
            "--output_json",
            left_json,
            *display_arg,
        ])
        run_step([
            python,
            "src/detect_right_hand_strings.py",
            "--video",
            str(video_path),
            "--output_json",
            right_json,
            *display_arg,
        ])

    if skip_vision_merge:
        if not Path(vision_json).exists():
            raise FileNotFoundError(f"Merged vision JSON not found: {vision_json}")
    else:
        run_step([
            python,
            "src/merge_hand_vision.py",
            "--left_json",
            left_json,
            "--right_json",
            right_json,
            "--out_json",
            vision_json,
        ])
    run_step([
        python,
        "src/run_tab_solver.py",
        "--audio_result_json",
        audio_json,
        "--vision_frames_json",
        vision_json,
        "--out_dir",
        solve_dir,
    ])
    run_step([
        python,
        "src/export_musicxml.py",
        "--solved_notes_json",
        str(Path(solve_dir) / "solved_notes.json"),
        "--audio_result_json",
        audio_json,
        "--out_musicxml",
        musicxml,
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/videos/test_video6.mp4")
    parser.add_argument("--audio_json", default="outputs/audio/result.json")
    parser.add_argument("--left_json", default="outputs/vision/left_frames.json")
    parser.add_argument("--right_json", default="outputs/vision/right_frames.json")
    parser.add_argument("--vision_json", default="outputs/vision/vision_frames.json")
    parser.add_argument("--solve_dir", default="outputs/solve")
    parser.add_argument("--musicxml", default="outputs/musicxml/tab_output.musicxml")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--no_demucs", action="store_true")
    parser.add_argument("--bpm_mode", choices=["combined", "drums", "bass"], default="combined")
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_vision_detection", action="store_true")
    parser.add_argument("--skip_vision_merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        video=args.video,
        audio_json=args.audio_json,
        left_json=args.left_json,
        right_json=args.right_json,
        vision_json=args.vision_json,
        solve_dir=args.solve_dir,
        musicxml=args.musicxml,
        no_display=args.no_display,
        use_demucs=not args.no_demucs,
        bpm_mode=args.bpm_mode,
        skip_audio=args.skip_audio,
        skip_vision_detection=args.skip_vision_detection,
        skip_vision_merge=args.skip_vision_merge,
    )
