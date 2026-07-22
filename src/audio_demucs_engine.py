import json
import os
import subprocess

import librosa
import numpy as np


def extract_bpm(audio_path):
    """
    Estimate BPM directly from an audio file.
    Returns (bpm, beat_times).
    """
    y, sr = librosa.load(audio_path, sr=None)
    y_percussive = librosa.effects.percussive(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, tightness=50)

    if isinstance(tempo, np.ndarray):
        bpm = float(tempo[0])
    else:
        bpm = float(tempo)

    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    return bpm, beat_times


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-8:
        return y
    return y / peak


def extract_bpm_from_stems(
    bass_path,
    drums_path,
    *,
    drums_weight: float = 0.7,
    bass_weight: float = 0.3,
    tightness: float = 80.0,
):
    """
    Estimate BPM by combining onset cues from both drums and bass stems.
    Drums lead the beat grid; bass reinforces downbeats and note attacks.
    Returns (bpm, beat_times).
    """
    y_drums, sr = librosa.load(drums_path, sr=None, mono=True)
    y_bass, sr_bass = librosa.load(bass_path, sr=sr, mono=True)

    if sr_bass != sr:
        raise ValueError("Bass/drums sample rates do not match after loading.")

    y_drums = _normalize_audio(y_drums)
    y_bass = _normalize_audio(y_bass)

    drums_percussive = librosa.effects.percussive(y_drums)
    _, bass_percussive = librosa.effects.hpss(y_bass)

    hop_length = 512
    drums_onset = librosa.onset.onset_strength(
        y=drums_percussive,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )
    bass_onset = librosa.onset.onset_strength(
        y=bass_percussive,
        sr=sr,
        hop_length=hop_length,
        lag=2,
        max_size=3,
        aggregate=np.mean,
    )

    max_len = max(len(drums_onset), len(bass_onset))
    drums_onset = librosa.util.fix_length(drums_onset, size=max_len)
    bass_onset = librosa.util.fix_length(bass_onset, size=max_len)

    drums_scale = np.max(drums_onset) or 1.0
    bass_scale = np.max(bass_onset) or 1.0
    combined_onset = (
        drums_weight * (drums_onset / drums_scale)
        + bass_weight * (bass_onset / bass_scale)
    )

    tempo, beats = librosa.beat.beat_track(
        onset_envelope=combined_onset,
        sr=sr,
        hop_length=hop_length,
        tightness=tightness,
    )

    if isinstance(tempo, np.ndarray):
        bpm = float(np.squeeze(tempo))
    else:
        bpm = float(tempo)

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length).tolist()
    return bpm, beat_times


class AutotabAudioEngine:
    def __init__(self, input_path, output_root="outputs/audio"):
        self.input_path = input_path
        self.output_root = output_root
        self.filename = os.path.splitext(os.path.basename(input_path))[0]

        os.makedirs(output_root, exist_ok=True)

        self.bass_wav = None
        self.drums_wav = None
        self.bpm = 0.0
        self.beat_times = []
        self.notes = []

    def run_demucs(self):
        print("\n[1/4] Demucs stem separation...")
        stem_root = os.path.join(self.output_root, "htdemucs", self.filename)
        self.bass_wav = os.path.join(stem_root, "bass.wav")
        self.drums_wav = os.path.join(stem_root, "drums.wav")

        if os.path.exists(self.bass_wav) and os.path.exists(self.drums_wav):
            print(f"Using existing stems: {stem_root}")
            return {"bass": self.bass_wav, "drums": self.drums_wav}

        command = f'demucs -d cpu -n htdemucs --shifts=10 "{self.input_path}" -o "{self.output_root}"'
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
        print("Stem separation finished")
        return {"bass": self.bass_wav, "drums": self.drums_wav}

    def analyze_bpm(self, bpm_mode="bass"):
        print("[2/4] BPM analysis...")
        if bpm_mode == "combined" and self.bass_wav and self.drums_wav:
            self.bpm, self.beat_times = extract_bpm_from_stems(self.bass_wav, self.drums_wav)
            print(f"Estimated BPM: {self.bpm:.2f} (bass + drums)")
            return

        if bpm_mode == "bass":
            bpm_source = self.bass_wav
        else:
            bpm_source = self.drums_wav or self.bass_wav

        self.bpm, self.beat_times = extract_bpm(bpm_source)
        print(f"Estimated BPM: {self.bpm:.2f} ({bpm_mode})")

    def extract_monophonic_notes(self):
        print("[3/4] Monophonic pitch extraction...")
        y, sr = librosa.load(self.bass_wav, sr=None)

        fmin = librosa.note_to_hz("C1")
        fmax = librosa.note_to_hz("C3")

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            fill_na=None,
        )

        times = librosa.times_like(f0, sr=sr)
        current_note = None

        for i, pitch in enumerate(f0):
            if pitch is not None and voiced_flag[i]:
                midi_pitch = int(round(librosa.hz_to_midi(pitch)))

                if current_note is None:
                    current_note = {
                        "start_s": times[i],
                        "pitch": midi_pitch,
                        "sum_prob": voiced_probs[i],
                        "count": 1,
                    }
                elif current_note["pitch"] == midi_pitch:
                    current_note["end_s"] = times[i]
                    current_note["sum_prob"] += voiced_probs[i]
                    current_note["count"] += 1
                else:
                    self._add_note(current_note)
                    current_note = {
                        "start_s": times[i],
                        "pitch": midi_pitch,
                        "sum_prob": voiced_probs[i],
                        "count": 1,
                    }
            elif current_note:
                self._add_note(current_note)
                current_note = None

        if current_note:
            self._add_note(current_note)

        print(f"Valid note events: {len(self.notes)}")

    def _add_note(self, note_dict):
        start = float(note_dict["start_s"])
        end = float(note_dict.get("end_s", start + 0.05))
        duration = end - start

        if duration > 0.05:
            self.notes.append(
                {
                    "start_s": round(start, 4),
                    "end_s": round(end, 4),
                    "pitch": int(note_dict["pitch"]),
                    "velocity": round(float(note_dict["sum_prob"] / note_dict["count"]), 2),
                }
            )

    def export_final_json(self):
        print("[4/4] Exporting JSON...")
        final_report = {
            "metadata": {
                "file": self.filename,
                "bpm": self.bpm,
                "bass_wav": self.bass_wav,
                "drums_wav": self.drums_wav,
            },
            "beat_grid": self.beat_times,
            "notes": self.notes,
        }

        json_path = os.path.join(self.output_root, f"{self.filename}_audio_final.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        print(f"Final audio report: {json_path}")
        return json_path

    def run(self, bpm_mode="bass"):
        self.run_demucs()
        self.analyze_bpm(bpm_mode=bpm_mode)
        self.extract_monophonic_notes()
        return self.export_final_json()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_audio", required=True)
    ap.add_argument("--output_root", default="outputs/audio")
    ap.add_argument("--bpm_mode", choices=["combined", "drums", "bass"], default="bass")
    args = ap.parse_args()

    engine = AutotabAudioEngine(args.input_audio, output_root=args.output_root)
    engine.run(bpm_mode=args.bpm_mode)
