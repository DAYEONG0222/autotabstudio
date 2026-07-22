# 프로젝트 구조 설명

이 프로젝트는 기타/베이스 연주 영상을 오디오 분석, 비전 분석, 결과 병합, TAB/MusicXML 변환 단계로 나누어 처리합니다.

## 1. 오디오 분석

- 실행 파일: `run_audio_analysis.py`
- 보조 모듈:
  - `src/audio/pyin_pitch_analyzer.py`
  - `src/audio/audio_note_segmenter.py`
  - `src/audio_demucs_engine.py`
- 주요 출력:
  - `outputs/audio/result.json`

오디오 분석 단계는 입력 영상 또는 오디오 파일에서 피치와 BPM을 추정하고, 연속적인 피치 프레임을 음표 단위 JSON으로 변환합니다. Demucs를 사용할 경우 베이스 stem을 분리한 뒤 분석할 수 있습니다.

## 2. 비전 분석

- 왼손 분석: `src/detect_left_hand_frets.py`
- 오른손 분석: `src/detect_right_hand_strings.py`
- 병합: `src/merge_hand_vision.py`
- 주요 출력:
  - `outputs/vision/left_frames.json`
  - `outputs/vision/right_frames.json`
  - `outputs/vision/vision_frames.json`

비전 분석 단계는 왼손 프렛 후보와 오른손 줄 후보를 프레임 단위로 추정합니다. 이후 두 결과를 시간 기준으로 맞춰 하나의 `vision_frames.json` 파일로 병합합니다.

## 3. 줄/프렛 추론

- 실행 파일: `src/run_tab_solver.py`
- 보조 모듈:
  - `src/audio_vision_adapter.py`
  - `src/bass_fingering_solver.py`
- 주요 출력:
  - `outputs/solve/vision_events.json`
  - `outputs/solve/solved_notes.json`
  - `outputs/solve/tab.txt`

이 단계는 오디오에서 검출된 음표와 비전 이벤트를 매칭해 각 음표의 베이스 줄과 프렛을 결정합니다.

## 4. MusicXML 내보내기

- 실행 파일: `src/export_musicxml.py`
- 주요 출력:
  - `outputs/musicxml/tab_output.musicxml`

`solved_notes.json`을 읽어 악보 프로그램에서 열 수 있는 MusicXML 파일을 생성합니다.

## 5. 파이프라인 실행 파일

- 실행 파일: `run_full_pipeline.py`

전체 분석을 한 번에 실행하거나, 이미 생성된 오디오/비전 JSON을 재사용해 일부 단계만 다시 실행할 수 있습니다.

주요 옵션:

- `--skip_audio`: 기존 오디오 JSON을 사용합니다.
- `--skip_vision_detection`: 기존 왼손/오른손 비전 JSON을 사용합니다.
- `--skip_vision_merge`: 기존 병합 비전 JSON을 사용합니다.
- `--no_display`: OpenCV 창을 띄우지 않고 처리합니다.
- `--no_demucs`: Demucs 분리를 사용하지 않습니다.

## 6. 폴더 역할

- `src/`: 실제 파이프라인에 가까운 핵심 코드
- `src/audio/`: 피치 분석과 음표 구간화 코드
- `src/preprocessing/`: 전처리 코드
- `src/legacy/`: 이전 구현 보관
- `experiments/`: 실험 및 프로토타입 코드
- `data/`: 로컬 영상, 이미지, 모델 가중치
- `outputs/`: 실행 결과 JSON, WAV, TAB, MusicXML
- `docs/`: 프로젝트 설명 문서

## 7. 검증 상태

2026-05-22 기준으로 기존 샘플 산출물을 사용해 다음 경로를 검증했습니다.

```text
left_frames.json + right_frames.json
-> vision_frames.json
-> solved_notes.json / tab.txt
-> tab_output.musicxml
-> music21 파싱 성공
```

검증 결과:

- 병합 비전 프레임 수: 700
- 최종 solved note 수: 226
- MusicXML 파싱 note 수: 233

원본 영상에서 오디오/비전 분석을 처음부터 다시 수행하는 경로는 처리 시간이 길 수 있으므로, 제출 전 시연에서는 샘플 산출물을 재사용하는 명령을 함께 제공하는 것을 권장합니다.
