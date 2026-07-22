import json
import os
import numpy as np
from music21 import stream, note, tempo, metadata, instrument

try:
    from music21.technical import FretIndication, StringIndication
except ImportError:
    from music21.articulations import FretIndication, StringIndication

# --- [1. 범용 엔진 설정 계수] ---
# 물리적/음악적 상수로, 대부분의 베이스 연주 환경에서 공통적으로 적용됩니다.
LATENCY_CORRECTION = -0.04 
COMFORT_FRET_RANGE = 5 
WINDOW_PENALTY = 100 

# 범용 줄 가중치: 특정 곡을 위한 극단적 수치가 아닌, 
# 베이스 기타의 일반적인 톤 선호도(굵은 줄 선호)만 살짝 반영합니다.
STRING_PREFERENCE = {4: 10, 3: 5, 2: 2, 1: 0} 

QUANTIZE_RESOLUTION = 0.25 
BASS_TUNING = {4: 28, 3: 33, 2: 38, 1: 43} 

BASE_DIR = r'C:\Autotab_Studio'
DATA_DIR = os.path.join(BASE_DIR, 'output')
AUDIO_JSON = os.path.join(DATA_DIR, 'test_video2_audio_final.json')
LEFT_HAND_JSON = os.path.join(DATA_DIR, 'left_hand_vision_log.json')
OUTPUT_XML = os.path.join(BASE_DIR, 'final_bass_tab_general.musicxml')

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def quantize(value, base=0.25):
    return round(float(value) / base) * base

def get_corrected_vision_data(logs, target_time):
    """지연 시간을 보정한 비전 데이터 추출 (범용 엔진 핵심)"""
    if not logs: return []
    corrected_time = target_time + LATENCY_CORRECTION
    closest_frame = min(logs, key=lambda x: abs(x['timestamp'] - corrected_time))
    if abs(closest_frame['timestamp'] - corrected_time) > 0.15: return []
    return closest_frame.get('detected_fingers', [])

# --- [2. 핵심 최적화 알고리즘] ---
def create_tab_xml():
    print(f" [General Engine] 범용 로직 기반 채보 시작...")
    
    audio_data = load_json(AUDIO_JSON)
    lh_logs = load_json(LEFT_HAND_JSON)
    if not audio_data: return

    bpm = audio_data['metadata'].get('bpm', 120.0)
    all_notes_candidates = []

    for audio_note in audio_data['notes']:
        raw_pitch = audio_note['pitch']
        lh_fingers = get_corrected_vision_data(lh_logs, audio_note['start_s'])
        
        # [범용 솔루션] 인트로 뮤트 시간 대신 '비전 존재 여부'로 판단
        # 손가락이 지판(Mask) 안에 감지되지 않는 시점의 소리는 무시합니다. [cite: 2026-03-15]
        if not any(f.get('in_mask', False) for f in lh_fingers):
            continue

        candidates = []
        for octave_shift in [0]: # 옥타브 고정 엔진
            test_pitch = raw_pitch + octave_shift
            for s_idx in [4, 3, 2, 1]:
                fret = test_pitch - BASS_TUNING[s_idx]
                if 0 <= fret <= 21:
                    v_bonus = 0
                    for lf in lh_fingers:
                        # 손가락 좌표와 후보 프렛/줄이 일치할 때 가점 부여
                        if lf.get('string') == s_idx and abs(lf.get('fret', -10) - fret) <= 1:
                            v_bonus = 45 * lf.get('confidence', 0.8)
                            break
                    
                    # 범용 가중치 합산
                    total_bonus = v_bonus + STRING_PREFERENCE[s_idx]
                    
                    candidates.append({
                        'string': s_idx, 'fret': fret, 'pitch': test_pitch, 
                        'bonus': total_bonus, 'raw_note': audio_note
                    })
        if candidates: all_notes_candidates.append(candidates)

    if not all_notes_candidates:
        print("💡 생성된 음표가 없습니다. 비전 데이터와 오디오 싱크를 확인하세요."); return

    # DP 최적화 (이동 거리 최소화 엔진)
    n_notes = len(all_notes_candidates)
    dp = [{} for _ in range(n_notes)]
    parent = [{} for _ in range(n_notes)]

    for j, cand in enumerate(all_notes_candidates[0]):
        dp[0][j] = -cand['bonus'] 

    for i in range(1, n_notes):
        for j, curr in enumerate(all_notes_candidates[i]):
            min_val, best_p = float('inf'), 0
            for p, prev in enumerate(all_notes_candidates[i-1]):
                diff = abs(curr['fret'] - prev['fret'])
                move_cost = diff * 0.8
                
                # 물리적 가용 범위 초과 시 패널티 (범용 규칙)
                if diff > COMFORT_FRET_RANGE:
                    move_cost += WINDOW_PENALTY 
                
                cost = dp[i-1][p] + move_cost - curr['bonus']
                if cost < min_val: min_val, best_p = cost, p
            dp[i][j], parent[i][j] = min_val, best_p

    # 경로 추적 및 MusicXML 생성
    path = [0] * n_notes
    curr_idx = min(dp[n_notes-1], key=dp[n_notes-1].get)
    path[n_notes-1] = curr_idx
    for i in range(n_notes-2, -1, -1): path[i] = parent[i+1][path[i+1]]

    s = stream.Score(); p = stream.Part(); p.id = 'Bass'
    p.insert(0, instrument.ElectricBass()); p.insert(0, tempo.MetronomeMark(number=bpm))
    
    for i, idx in enumerate(path):
        res = all_notes_candidates[i][idx]
        a_n = res['raw_note']
        q_off = quantize(a_n['start_s'] * bpm / 60, QUANTIZE_RESOLUTION)
        q_dur = max(0.25, quantize((a_n['end_s']-a_n['start_s']) * bpm / 60, QUANTIZE_RESOLUTION))
        
        n = note.Note(res['pitch']) 
        n.duration.quarterLength = q_dur
        n.articulations.extend([FretIndication(int(res['fret'])), StringIndication(int(res['string']))])
        p.insert(q_off, n)

    s.append(p.makeMeasures())
    s.insert(0, metadata.Metadata(title='AI Bass Tab (General Engine)', composer='Autotab Studio'))
    s.write('musicxml', fp=OUTPUT_XML)
    print(f"범용 엔진 채보 완료: {OUTPUT_XML}")

if __name__ == "__main__":
    create_tab_xml()