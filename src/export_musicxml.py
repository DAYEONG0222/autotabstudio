# export_musicxml.py
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

# requests 경고 숨김
try:
    from requests import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seconds_to_quarter_length(seconds: float, bpm: float) -> float:
    if bpm <= 0:
        return 1.0
    sec_per_quarter = 60.0 / bpm
    return max(0.0, float(seconds / sec_per_quarter))


def quantize_position(value: float, grid: float) -> float:
    if grid <= 0:
        return float(value)
    return round(float(value) / grid) * grid


def quantize_quarter_length(ql: float) -> float:
    """
    MusicXML에서 안전하게 표현할 수 있는 기본 음가로 양자화.
    """
    allowed = [
        4.0,   # whole
        3.0,   # dotted half
        2.0,   # half
        1.5,   # dotted quarter
        1.0,   # quarter
        0.75,  # dotted eighth
        0.5,   # eighth
        0.25,  # 16th
    ]
    return min(allowed, key=lambda x: abs(x - ql))


def split_into_expressible_lengths(ql: float) -> List[float]:
    """
    임의의 quarterLength를 MusicXML에서 표현 가능한 길이들의 합으로 분해.
    greedy 방식.
    """
    allowed = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]

    if ql <= 0:
        return []

    # 너무 짧은 값은 최소 16분음표로 처리
    if ql < 0.25:
        return [0.25]

    out = []
    remaining = ql

    while remaining >= 0.25:
        chosen = None
        for a in allowed:
            if a <= remaining + 1e-6:
                chosen = a
                break
        if chosen is None:
            break
        out.append(chosen)
        remaining -= chosen

        # 부동소수점 오차 정리
        if abs(remaining) < 1e-6:
            remaining = 0.0
            break

    # 남은 조각이 있으면 가장 가까운 표현 가능 음가 하나로 마무리
    if remaining > 0.0:
        out.append(quantize_quarter_length(remaining))

    return out


def quantize_solved_notes_to_grid(
    solved_notes: List[Dict[str, Any]],
    bpm: float,
    grid_ql: float,
    min_note_ql: float,
) -> List[Dict[str, Any]]:
    if not solved_notes:
        return []

    quantized = []
    prev_end = 0.0

    for note_data in sorted(solved_notes, key=lambda x: float(x["t_on"])):
        t_on = seconds_to_quarter_length(float(note_data["t_on"]), bpm)
        t_off = seconds_to_quarter_length(float(note_data["t_off"]), bpm)

        q_on = quantize_position(t_on, grid_ql)
        q_off = quantize_position(t_off, grid_ql)

        if q_on < prev_end:
            q_on = prev_end

        if q_off <= q_on:
            q_off = q_on + min_note_ql

        q_dur = max(min_note_ql, q_off - q_on)
        q_off = q_on + q_dur

        quantized.append({
            **note_data,
            "t_on_ql": q_on,
            "t_off_ql": q_off,
            "dur_ql": q_dur,
        })
        prev_end = q_off

    return quantized


def _load_music21():
    from music21 import expressions, instrument, metadata, meter, note, stream, tempo

    return {
        "expressions": expressions,
        "instrument": instrument,
        "metadata": metadata,
        "meter": meter,
        "note": note,
        "stream": stream,
        "tempo": tempo,
    }


# 내부 줄 번호는 1=E(가장 낮은 줄) ... 4=G 이다.
STRING_NAMES = {1: "E", 2: "A", 3: "D", 4: "G"}


def to_musicxml_string(string_num: int, num_strings: int = 4) -> int:
    """MusicXML은 가장 높은 줄을 1번으로 센다(4현 베이스: 1=G ... 4=E)."""
    return num_strings + 1 - int(string_num)


def add_string_fret_lyrics(n, string_num: int, fret_num: int):
    n.lyric = f"{STRING_NAMES[string_num]} {fret_num}"


def export_basic_musicxml_with_tab_hints(
    solved_notes: List[Dict[str, Any]],
    bpm: float,
    out_path: str,
    title: str = "AutoTabStudio Output",
    composer: str = "AutoTabStudio",
    time_signature: str = "4/4",
    rhythm_grid: float = 0.5,
) -> str:
    music21 = _load_music21()
    expressions = music21["expressions"]
    instrument = music21["instrument"]
    metadata = music21["metadata"]
    meter = music21["meter"]
    note = music21["note"]
    stream = music21["stream"]
    tempo = music21["tempo"]

    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title
    sc.metadata.composer = composer

    part = stream.Part()
    part.id = "BassPart"
    part.insert(0, instrument.ElectricBass())
    part.append(tempo.MetronomeMark(number=float(bpm)))
    part.append(meter.TimeSignature(time_signature))

    sorted_notes = quantize_solved_notes_to_grid(
        solved_notes=solved_notes,
        bpm=bpm,
        grid_ql=rhythm_grid,
        min_note_ql=rhythm_grid,
    )

    prev_t_off = 0.0

    for sn in sorted_notes:
        t_on = float(sn["t_on_ql"])
        t_off = float(sn["t_off_ql"])
        pitch = int(sn["pitch"])
        string_num = int(sn["string"])
        fret_num = int(sn["fret"])

        # 1) rest 삽입
        if t_on > prev_t_off:
            rest_ql_raw = t_on - prev_t_off
            rest_parts = split_into_expressible_lengths(rest_ql_raw)

            for rql in rest_parts:
                r = note.Rest()
                r.duration.quarterLength = rql
                part.append(r)

        # 2) note 삽입
        note_ql_raw = max(rhythm_grid, t_off - t_on)
        note_parts = split_into_expressible_lengths(note_ql_raw)

        # 음표가 너무 잘게 쪼개져 사라지는 것을 막기 위해 최소 하나는 보장
        if not note_parts:
            note_parts = [rhythm_grid]

        for i, ql in enumerate(note_parts):
            n = note.Note(pitch)
            n.duration.quarterLength = ql

            # 첫 음표 조각에만 string/fret 정보 표기
            if i == 0:
                add_string_fret_lyrics(n, string_num, fret_num)
                txt = expressions.TextExpression(f"str {STRING_NAMES[string_num]} fret {fret_num}")
                n.expressions.append(txt)

            part.append(n)

        prev_t_off = max(prev_t_off, t_off)

    sc.append(part)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sc.write("musicxml", fp=out_path)
    return out_path


def inject_tab_technical_to_musicxml(
    solved_notes: List[Dict[str, Any]],
    input_musicxml_path: str,
    output_musicxml_path: str
) -> str:
    import xml.etree.ElementTree as ET

    tree = ET.parse(input_musicxml_path)
    root = tree.getroot()

    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].strip("{")
        ns = {"m": ns_uri}
        note_xpath = ".//m:note"
    else:
        ns = {}
        note_xpath = ".//note"

    solved_idx = 0

    for note_elem in root.findall(note_xpath, ns):
        rest_elem = note_elem.find("m:rest", ns) if ns else note_elem.find("rest")
        pitch_elem = note_elem.find("m:pitch", ns) if ns else note_elem.find("pitch")
        if rest_elem is not None or pitch_elem is None:
            continue

        if solved_idx >= len(solved_notes):
            break

        solved = solved_notes[solved_idx]

        # lyric/text가 붙은 '첫 음표 조각'에만 technical을 넣도록
        # lyric이 있는 note만 index 증가
        lyric_elem = note_elem.find("m:lyric", ns) if ns else note_elem.find("lyric")
        if lyric_elem is None:
            continue

        string_num = int(solved["string"])
        fret_num = int(solved["fret"])

        notations = note_elem.find("m:notations", ns) if ns else note_elem.find("notations")
        if notations is None:
            notations = ET.SubElement(note_elem, "notations")

        technical = None
        for child in list(notations):
            tag_name = child.tag.split("}")[-1]
            if tag_name == "technical":
                technical = child
                break

        if technical is None:
            technical = ET.SubElement(notations, "technical")

        string_el = ET.SubElement(technical, "string")
        string_el.text = str(to_musicxml_string(string_num))

        fret_el = ET.SubElement(technical, "fret")
        fret_el.text = str(fret_num)

        solved_idx += 1

    Path(output_musicxml_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_musicxml_path, encoding="utf-8", xml_declaration=True)
    return output_musicxml_path


def export_tab_musicxml(
    solved_notes_json: str,
    audio_result_json: Optional[str],
    out_musicxml_path: str = "outputs/musicxml/tab_output.musicxml",
    title: str = "AutoTabStudio TAB",
    composer: str = "AutoTabStudio",
    time_signature: str = "4/4",
    rhythm_grid: float = 0.5,
) -> Dict[str, Any]:
    solved_notes = load_json(solved_notes_json)

    bpm = 120.0
    if audio_result_json is not None:
        audio_data = load_json(audio_result_json)
        bpm = float(audio_data.get("bpm", 120.0))

    temp_path = str(Path(out_musicxml_path).with_name(Path(out_musicxml_path).stem + "_temp.musicxml"))

    basic_path = export_basic_musicxml_with_tab_hints(
        solved_notes=solved_notes,
        bpm=bpm,
        out_path=temp_path,
        title=title,
        composer=composer,
        time_signature=time_signature,
        rhythm_grid=rhythm_grid,
    )

    final_path = inject_tab_technical_to_musicxml(
        solved_notes=solved_notes,
        input_musicxml_path=basic_path,
        output_musicxml_path=out_musicxml_path,
    )

    return {
        "bpm": bpm,
        "input_solved_notes_json": solved_notes_json,
        "input_audio_result_json": audio_result_json,
        "temp_musicxml": basic_path,
        "final_musicxml": final_path,
        "note_count": len(solved_notes),
        "rhythm_grid": rhythm_grid,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--solved_notes_json", default="outputs/solve/solved_notes.json")
    ap.add_argument("--audio_result_json", default=None)
    ap.add_argument("--out_musicxml", default="outputs/musicxml/tab_output.musicxml")
    ap.add_argument("--title", default="AutoTabStudio TAB")
    ap.add_argument("--composer", default="AutoTabStudio")
    ap.add_argument("--time_signature", default="4/4")
    ap.add_argument("--rhythm_grid", type=float, default=0.5)

    args = ap.parse_args()

    result = export_tab_musicxml(
        solved_notes_json=args.solved_notes_json,
        audio_result_json=args.audio_result_json,
        out_musicxml_path=args.out_musicxml,
        title=args.title,
        composer=args.composer,
        time_signature=args.time_signature,
        rhythm_grid=args.rhythm_grid,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
