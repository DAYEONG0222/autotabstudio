import librosa
import numpy as np
from pathlib import Path


def _estimate_raw_bpm(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """HPSS percussive 기반 BPM 추정 내부 함수"""
    _, y_percussive = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    return float(np.squeeze(tempo)), beats


def _correct_octave(bpm_target: float, bpm_ref: float) -> float:
    """
    bpm_ref(bass 단독)를 기준으로 bpm_target(bass_drum)의 옥타브 오류 보정.

    bpm_target이 bpm_ref의 절반 근처 → × 2
    bpm_target이 bpm_ref의 두 배 근처 → ÷ 2
    그 외 → 그대로 사용
    """
    ratio = bpm_target / bpm_ref

    if 0.4 < ratio < 0.6:       # 절반 근처 (예: 83 vs 166)
        corrected = bpm_target * 2
        print(f"  [옥타브 보정] {bpm_target:.2f} × 2 → {corrected:.2f}  (절반 템포 감지)")
        return corrected

    elif 1.8 < ratio < 2.2:     # 두 배 근처 (예: 332 vs 166)
        corrected = bpm_target / 2
        print(f"  [옥타브 보정] {bpm_target:.2f} ÷ 2 → {corrected:.2f}  (두 배 템포 감지)")
        return corrected

    else:
        print(f"  [옥타브 보정] 보정 불필요 (ratio: {ratio:.2f})")
        return bpm_target


def estimate_bpm(bass_path: Path, bass_drum_path: Path) -> tuple[float, list[float]]:
    """
    bass.wav와 bass_drum.wav를 비교하여 옥타브 보정된 BPM과 비트 타임스탬프 반환.

    Args:
        bass_path      : bass.wav 경로 (옥타브 보정 기준)
        bass_drum_path : bass_drum.wav 경로 (최종 BPM 및 beat_times 산출)

    Returns:
        bpm        : 보정된 BPM (float)
        beat_times : 비트 타임스탬프 리스트 (초 단위)
    """
    for path in [bass_path, bass_drum_path]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    print(f"\n{'=' * 60}")
    print(f"BPM 추정")
    print(f"{'=' * 60}")

    # 1. bass 단독 BPM (보정 기준)
    print(f"[1/2] bass 단독 BPM 추정 중: {bass_path.name}")
    y_bass, sr_bass = librosa.load(str(bass_path), mono=True)
    bpm_bass, _ = _estimate_raw_bpm(y_bass, sr_bass)
    print(f"      bass BPM : {bpm_bass:.2f}")

    # 2. bass_drum BPM 추정 + beat_times 산출
    print(f"[2/2] bass_drum BPM 추정 중: {bass_drum_path.name}")
    y_bd, sr_bd = librosa.load(str(bass_drum_path), mono=True)
    bpm_bd, beats = _estimate_raw_bpm(y_bd, sr_bd)
    print(f"      bass_drum BPM : {bpm_bd:.2f}")

    # 3. 옥타브 보정
    print(f"\n=== 옥타브 보정 ===")
    bpm_final = _correct_octave(bpm_target=bpm_bd, bpm_ref=bpm_bass)

    beat_times = librosa.frames_to_time(beats, sr=sr_bd).tolist()

    print(f"\n=== BPM 추정 결과 ===")
    print(f"  bass 단독 BPM    : {bpm_bass:.2f}")
    print(f"  bass_drum BPM    : {bpm_bd:.2f}  (보정 전)")
    print(f"  최종 BPM         : {bpm_final:.2f}  ← 채택")
    print(f"\n  감지된 비트 수   : {len(beats)}개")
    if beat_times:
        print(f"  첫 번째 비트     : {beat_times[0]:.2f}초")
        print(f"  마지막 비트      : {beat_times[-1]:.2f}초")
    print(f"{'=' * 60}\n")

    return bpm_final, beat_times


if __name__ == "__main__":
    bass_path      = Path(r"C:\Users\admin\Documents\GitHub\MyProject\Tab_CV_Bass\data\output\bass.wav")
    bass_drum_path = Path(r"C:\Users\admin\Documents\GitHub\MyProject\Tab_CV_Bass\data\output\bass_drum.wav")
    bpm, beat_times = estimate_bpm(bass_path, bass_drum_path)