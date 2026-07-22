import subprocess
from pathlib import Path

# audio 추출
def extract_audio_from_video(
    input_video: Path,
    output_audio: Path,
    sr: int = 48000, # 영상 기준 48kHz
    mono: bool = True,
):
    """
    영상 파일에서 오디오만 추출해서 wav로 저장

    Args:
        input_video: 입력 비디오 파일 경로
        output_audio: 출력 오디오 파일 경로 (.wav)
        sr: 샘플레이트 (CD 기준 : 44100Hz)
        mono: 모노로 변환 여부 (기본: True)
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # 덮어쓰기
        "-i", str(input_video),
        "-vn",  # 비디오 스트림 제거
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sr),
    ]

    if mono:
        cmd += ["-ac", "1"]  # 모노
    else:
        cmd += ["-ac", "2"]  # 스테레오

    cmd.append(str(output_audio))

    print(f"[ffmpeg] 오디오 추출 중...")
    print(f"  입력: {input_video}")
    print(f"  출력: {output_audio}")

    subprocess.run(cmd, check=True)
    print(f"[완료] 오디오 추출 완료")

# video 추출
def extract_video_without_audio(
    input_video: Path,
    output_video: Path,
):
    """
    영상 파일에서 오디오를 제거하고 비디오만 저장

    Args:
        input_video: 입력 비디오 파일 경로
        output_video: 출력 비디오 파일 경로
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-an",  # 오디오 스트림 제거
        "-c:v", "copy",  # 비디오 코덱은 복사 (재인코딩 안 함)
        str(output_video),
    ]

    print(f"[ffmpeg] 비디오 추출 중...")
    print(f"  입력: {input_video}")
    print(f"  출력: {output_video}")

    subprocess.run(cmd, check=True)
    print(f"[완료] 비디오 추출 완료")


def separate_av(
    input_video: Path,
    output_audio: Path,
    output_video: Path,
    sample_rate: int = 44100,
    mono: bool = True,
):
    """
    동영상을 오디오와 비디오로 분리

    Args:
        input_video: 입력 비디오 파일 경로
        output_audio: 출력 오디오 파일 경로
        output_video: 출력 비디오 파일 경로
        sample_rate: 오디오 샘플레이트
        mono: 모노로 변환 여부
    """
    if not input_video.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없음: {input_video}")

    print(f"\n{'=' * 60}")
    print(f"Audio/Video Separation")
    print(f"{'=' * 60}")
    print(f"입력 파일: {input_video.name}")

    # 오디오 추출
    extract_audio_from_video(input_video, output_audio, sample_rate, mono)

    # 비디오 추출
    extract_video_without_audio(input_video, output_video)

    print(f"\n{'=' * 60}")
    print(f"분리 완료!")
    print(f"  오디오: {output_audio}")
    print(f"  비디오: {output_video}")
    print(f"{'=' * 60}\n")