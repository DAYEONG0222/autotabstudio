"""기준 TAB과 파이프라인 결과를 음 단위로 비교한다.

정답 기준: 시작 시각이 허용 오차 안에서 짝지어진 음 중 피치·줄·프렛이 모두 같은 음.
기준 악보의 시각(화면 커서 기준)과 오디오 시각 사이의 고정 지연은 전체 오프셋을 탐색해 맞춘다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reference(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = data["notes"] if isinstance(data, dict) else data
    return sorted(
        ({"t": float(n["onset"]), "midi": int(n["midi"]), "string": int(n["string"]), "fret": int(n["fret"])} for n in notes),
        key=lambda n: n["t"],
    )


def load_solved(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = data["notes"] if isinstance(data, dict) else data
    return sorted(
        ({"t": float(n["t_on"]), "midi": int(n["pitch"]), "string": n.get("string"), "fret": n.get("fret")} for n in notes),
        key=lambda n: n["t"],
    )


def load_audio_notes(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(
        ({"t": float(n["time_sec"]), "midi": int(n["pitch"]), "string": None, "fret": None} for n in data["notes"]),
        key=lambda n: n["t"],
    )


def match_onsets(ref: list[dict], pred: list[dict], offset: float, tol: float) -> list[tuple[int, int]]:
    """시각 차이가 작은 쌍부터 1:1로 짝짓는다."""
    pairs = []
    j0 = 0
    for i, r in enumerate(ref):
        t = r["t"] + offset
        while j0 < len(pred) and pred[j0]["t"] < t - tol:
            j0 += 1
        j = j0
        while j < len(pred) and pred[j]["t"] <= t + tol:
            pairs.append((abs(pred[j]["t"] - t), i, j))
            j += 1
    pairs.sort()
    used_r, used_p, matched = set(), set(), []
    for _, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        matched.append((i, j))
    return matched


def best_offset(ref: list[dict], pred: list[dict], tol: float, max_offset: float, step: float = 0.01) -> float:
    best, best_n = 0.0, -1
    n_steps = int(round(max_offset / step))
    for k in sorted(range(-n_steps, n_steps + 1), key=abs):
        off = k * step
        n = len(match_onsets(ref, pred, off, tol))
        if n > best_n:
            best, best_n = off, n
    return best


def evaluate(ref: list[dict], pred: list[dict], tol: float = 0.1, max_offset: float = 2.0, offset: float | None = None) -> dict:
    if offset is None:
        offset = best_offset(ref, pred, tol, max_offset)
    matched = match_onsets(ref, pred, offset, tol)
    n_ref, n_pred, n_match = len(ref), len(pred), len(matched)
    pitch_ok = [(i, j) for i, j in matched if ref[i]["midi"] == pred[j]["midi"]]
    pitch_class_ok = [(i, j) for i, j in matched if (ref[i]["midi"] - pred[j]["midi"]) % 12 == 0]
    full_ok = [
        (i, j) for i, j in pitch_ok
        if pred[j]["string"] == ref[i]["string"] and pred[j]["fret"] == ref[i]["fret"]
    ]

    def ratio(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    precision, recall = ratio(n_match, n_pred), ratio(n_match, n_ref)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "offset_sec": round(offset, 3),
        "tolerance_sec": tol,
        "reference_notes": n_ref,
        "predicted_notes": n_pred,
        "onset_matched": n_match,
        "onset_precision": precision,
        "onset_recall": recall,
        "onset_f1": f1,
        "pitch_correct": len(pitch_ok),
        "pitch_class_correct": len(pitch_class_ok),
        "full_correct": len(full_ok),
        # 정답률: 기준 악보 음 중 시각·피치·줄·프렛이 모두 맞은 비율
        "note_accuracy": ratio(len(full_ok), n_ref),
        "pitch_accuracy": ratio(len(pitch_ok), n_ref),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare solved TAB notes with a reference TAB.")
    parser.add_argument("--reference", required=True, help="reference JSON (notes: onset, string, fret, midi)")
    parser.add_argument("--solved", help="solved_notes.json from run_tab_solver.py")
    parser.add_argument("--audio", help="audio result JSON (pitch-only evaluation of audio notes)")
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max_offset", type=float, default=2.0)
    parser.add_argument("--out_json")
    args = parser.parse_args()

    ref = load_reference(args.reference)
    report = {"reference": str(args.reference)}
    if args.audio:
        audio = load_audio_notes(args.audio)
        report["audio"] = evaluate(ref, audio, args.tolerance, args.max_offset)
    if args.solved:
        solved = load_solved(args.solved)
        # 오디오 음표와 같은 오프셋을 쓰도록 오디오 결과가 있으면 그 값을 재사용한다.
        off = report["audio"]["offset_sec"] if "audio" in report else None
        report["solved"] = evaluate(ref, solved, args.tolerance, args.max_offset, offset=off)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
