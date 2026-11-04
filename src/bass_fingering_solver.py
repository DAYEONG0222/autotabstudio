# guitar_note_solver.py
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


# 4현 베이스 표준 튜닝(E1-A1-D2-G2). 기타 아래 4줄(E2-A2-D3-G3)보다 한 옥타브 낮다.
DEFAULT_OPEN_STRING_MIDI = {
    1: 28,  # E1
    2: 33,  # A1
    3: 38,  # D2
    4: 43,  # G2
}


@dataclass
class SolvedNote:
    t_on: float
    t_off: float
    pitch: int
    string: int
    fret: int
    audio_conf: float
    vision_conf: float
    total_score: float
    source_event_time: Optional[float]


def normalize_candidates(event: Dict[str, Any]):
    t = float(event["t"])

    if "string_candidates" in event:
        string_cands = [
            {"string": int(x["string"]), "conf": float(x.get("conf", 1.0))}
            for x in event["string_candidates"]
        ]
    elif "string" in event:
        string_cands = [{"string": int(event["string"]), "conf": float(event.get("conf", 1.0))}]
    else:
        string_cands = []

    if "fret_candidates" in event:
        fret_cands = [
            {"fret": int(x["fret"]), "conf": float(x.get("conf", 1.0))}
            for x in event["fret_candidates"]
        ]
    elif "fret" in event:
        fret_cands = [{"fret": int(event["fret"]), "conf": float(event.get("conf", 1.0))}]
    else:
        fret_cands = []

    return string_cands, fret_cands, t


def candidate_positions_for_pitch(
    pitch: int,
    open_string_midi: Dict[int, int],
    max_fret: int = 24
) -> List[Tuple[int, int]]:
    out = []
    for string_id, open_midi in open_string_midi.items():
        fret = pitch - open_midi
        if 0 <= fret <= max_fret:
            out.append((string_id, fret))
    return out


def find_nearest_vision_event(
    t_on: float,
    vision_events: List[Dict[str, Any]],
    time_gate: float = 0.15
) -> Optional[Dict[str, Any]]:
    best = None
    best_dt = float("inf")
    for ev in vision_events:
        dt = abs(float(ev["t"]) - t_on)
        if dt <= time_gate and dt < best_dt:
            best = ev
            best_dt = dt
    return best


def build_conf_map_string(string_cands: List[Dict[str, float]]) -> Dict[int, float]:
    out = {}
    for c in string_cands:
        out[int(c["string"])] = max(out.get(int(c["string"]), 0.0), float(c["conf"]))
    return out


def build_conf_map_fret(fret_cands: List[Dict[str, float]]) -> Dict[int, float]:
    out = {}
    for c in fret_cands:
        out[int(c["fret"])] = max(out.get(int(c["fret"]), 0.0), float(c["conf"]))
    return out


def score_position(
    t_on: float,
    pos: Tuple[int, int],
    event: Optional[Dict[str, Any]],
    prev_pos: Optional[Tuple[int, int]],
    audio_conf: float,
    w_string: float = 2.0,
    w_fret: float = 2.5,
    w_time: float = 1.0,
    w_audio: float = 1.0,
    w_transition: float = 0.20,
):
    string_id, fret = pos
    score = w_audio * float(audio_conf)
    vision_conf_component = 0.0

    if event is not None:
        string_cands, fret_cands, ev_t = normalize_candidates(event)
        string_map = build_conf_map_string(string_cands)
        fret_map = build_conf_map_fret(fret_cands)

        s_conf = string_map.get(string_id, 0.0)
        f_conf = fret_map.get(fret, 0.0)
        dt = abs(t_on - ev_t)

        score += w_string * s_conf
        score += w_fret * f_conf
        score -= w_time * dt

        vision_conf_component = 0.5 * (s_conf + f_conf)

    if prev_pos is not None:
        prev_string, prev_fret = prev_pos
        delta_string = abs(string_id - prev_string)
        delta_fret = abs(fret - prev_fret)
        score -= w_transition * (0.8 * delta_string + 1.0 * delta_fret)

    return score, vision_conf_component


def solve_notes_greedy(
    audio_notes: List[Dict[str, Any]],
    vision_events: List[Dict[str, Any]],
    open_string_midi: Dict[int, int] = DEFAULT_OPEN_STRING_MIDI,
    max_fret: int = 24,
    time_gate: float = 0.15,
) -> List[SolvedNote]:
    solved: List[SolvedNote] = []
    prev_pos: Optional[Tuple[int, int]] = None

    audio_notes_sorted = sorted(audio_notes, key=lambda x: float(x["t_on"]))

    for note in audio_notes_sorted:
        t_on = float(note["t_on"])
        t_off = float(note.get("t_off", t_on + 0.1))
        pitch = int(note["pitch"])
        audio_conf = float(note.get("conf", 1.0))

        positions = candidate_positions_for_pitch(
            pitch=pitch,
            open_string_midi=open_string_midi,
            max_fret=max_fret,
        )
        if not positions:
            continue

        event = find_nearest_vision_event(t_on, vision_events, time_gate=time_gate)

        best_score = -1e18
        best_pos = None
        best_vconf = 0.0
        best_event_time = None if event is None else float(event["t"])

        for pos in positions:
            score, vconf = score_position(
                t_on=t_on,
                pos=pos,
                event=event,
                prev_pos=prev_pos,
                audio_conf=audio_conf,
            )
            if score > best_score:
                best_score = score
                best_pos = pos
                best_vconf = vconf

        if best_pos is None:
            continue

        string_id, fret = best_pos
        solved.append(
            SolvedNote(
                t_on=t_on,
                t_off=t_off,
                pitch=pitch,
                string=string_id,
                fret=fret,
                audio_conf=audio_conf,
                vision_conf=best_vconf,
                total_score=best_score,
                source_event_time=best_event_time,
            )
        )
        prev_pos = best_pos

    return solved


def has_onset_near(t: float, onsets: List[float], tol: float) -> bool:
    i = bisect_left(onsets, t - tol)
    return i < len(onsets) and onsets[i] <= t + tol


def merge_same_position_notes(
    solved_notes: List[SolvedNote],
    max_gap: float = 0.05,
    onsets: Optional[List[float]] = None,
    onset_tol: float = 0.06,
) -> List[SolvedNote]:
    """
    Join pieces of one sustained note that pitch tracking split apart.
    With onset times, a note that starts at a detected attack is a re-played
    note (e.g. repeated eighth notes) and is kept separate.
    """
    if not solved_notes:
        return []

    notes = sorted(solved_notes, key=lambda x: x.t_on)
    merged = [notes[0]]

    for cur in notes[1:]:
        prev = merged[-1]
        same_position = (
            prev.pitch == cur.pitch and
            prev.string == cur.string and
            prev.fret == cur.fret
        )
        small_gap = (cur.t_on - prev.t_off) <= max_gap
        re_attacked = onsets is not None and has_onset_near(cur.t_on, onsets, onset_tol)

        if same_position and small_gap and not re_attacked:
            prev.t_off = max(prev.t_off, cur.t_off)
            prev.audio_conf = max(prev.audio_conf, cur.audio_conf)
            prev.vision_conf = max(prev.vision_conf, cur.vision_conf)
            prev.total_score = max(prev.total_score, cur.total_score)
        else:
            merged.append(cur)

    return merged


def solved_notes_to_dicts(solved_notes: List[SolvedNote]) -> List[Dict[str, Any]]:
    return [
        {
            "t_on": n.t_on,
            "t_off": n.t_off,
            "pitch": n.pitch,
            "string": n.string,
            "fret": n.fret,
            "audio_conf": n.audio_conf,
            "vision_conf": n.vision_conf,
            "total_score": n.total_score,
            "source_event_time": n.source_event_time,
        }
        for n in solved_notes
    ]


def export_ascii_tab(solved_notes: List[SolvedNote], out_path: str) -> str:
    if not solved_notes:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("G|\nD|\nA|\nE|\n")
        return out_path

    notes = sorted(solved_notes, key=lambda x: x.t_on)
    unique_times = sorted(set(round(n.t_on, 4) for n in notes))
    time_to_col = {t: i for i, t in enumerate(unique_times)}

    cols = len(unique_times)
    lines = [["---"] * cols for _ in range(4)]

    for n in notes:
        col = time_to_col[round(n.t_on, 4)]
        row = 4 - n.string  # 4->0(G), 3->1(D), 2->2(A), 1->3(E)
        cell = str(n.fret)
        if len(cell) == 1:
            cell = "-" + cell + "-"
        elif len(cell) == 2:
            cell = cell + "-"
        else:
            cell = cell[:3]
        lines[row][col] = cell

    out_lines = []
    for name, row in zip(["G", "D", "A", "E"], lines):
        out_lines.append(name + "|" + "".join(row) + "|")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    return out_path
