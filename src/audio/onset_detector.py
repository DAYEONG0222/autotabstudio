from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf


def detect_onsets(audio_path: Path, hop_length: int = 512) -> List[float]:
    """
    Detect note attack times (seconds) from the bass stem.
    pYIN only tracks pitch, so the same note played twice in a row can look like
    one long note. Onset times let later steps tell a re-attack from a sustain.
    """
    audio, sr = sf.read(str(audio_path))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        units="time",
    )

    print("[Onset detection] finished")
    print(f"  onsets: {len(onsets)}")
    return [round(float(t), 4) for t in onsets]
