# AutoTabStudio · 연주 영상 기반 악보 자동 생성

> 오디오 피치 분석과 비전 손 위치 분석을 결합해, 베이스 연주 영상을 TAB·MusicXML 악보로 자동 변환하는 멀티모달 시스템

**2025 캡스톤디자인 2** · 4인 팀 프로젝트 (2025.09 – 2026.03) · 🎯 악보 생성 정확도 **59% → 85%**

`Python` · `librosa (pYIN)` · `Demucs` · `YOLO` · `MediaPipe` · `OpenCV` · `music21`

---

## 개요

기타·베이스는 **같은 음을 여러 줄과 프렛에서** 연주할 수 있어, 오디오만으로는 실제 운지 위치를 알 수 없습니다. 반대로 영상만으로는 정확한 피치와 음의 시작 지점을 안정적으로 얻기 어렵습니다.

AutoTabStudio는 **오디오는 피치·타이밍을, 영상은 줄·프렛 후보를** 담당하도록 역할을 나눈 뒤, 두 결과를 **시간축 기준으로 병합**해 최종 운지를 추론합니다. 결과는 ASCII TAB과 MusicXML로 내보내 MuseScore 같은 악보 프로그램에서 바로 열 수 있습니다.

## 파이프라인

```mermaid
flowchart LR
    video["입력 영상"] --> audio["오디오 분석<br/>Demucs stem 분리<br/>pYIN 피치·BPM<br/>음표 구간화"]
    video --> vision["비전 분석<br/>YOLO · MediaPipe<br/>왼손 프렛 후보<br/>오른손 줄 후보"]
    audio --> merge["시간축 병합<br/>음표 타이밍 매칭<br/>줄/프렛 후보 점수화<br/>최종 운지 추론"]
    vision --> merge
    merge --> out["악보 생성<br/>ASCII TAB · MusicXML"]
```

## 핵심 구현

**1. 오디오 분석 파이프라인** — 전체 음원에서 바로 피치를 추정하면 반주·노이즈의 영향을 받기 때문에, **Demucs로 베이스 stem을 분리한 뒤** 분석합니다. librosa 기반 **pYIN**으로 시간 프레임별 피치를 추정하고 BPM·음표 시작·종료를 계산해, 연속 피치 흐름을 음표 단위 JSON 타임라인으로 저장합니다. 이 타임라인이 이후 비전 결과와 매칭되는 기준이 됩니다.

**2. 비전 분석과 결과 병합** — 영상 프레임에서 YOLO 객체 탐지와 MediaPipe 손 좌표로 왼손 프렛 후보·오른손 줄 후보를 프레임별로 저장하고, 별도 JSON으로 생성되는 좌/우손 결과를 시간 기준으로 정렬해 하나의 `vision_frames.json`으로 병합합니다.

**3. 운지 추론과 악보 생성** — 같은 시간대의 오디오(음높이·타이밍)와 비전(손 위치) 결과를 비교하고, 베이스 튜닝 규칙으로 가능한 줄/프렛 조합을 점수화해 최종 운지를 추론합니다. 결과를 실제 사용 가능한 악보 산출물로 변환합니다.

## 커스텀 YOLO 데이터셋 구축

공개 데이터셋이 없어 유튜브·핀터레스트 등에서 기타·베이스 연주 이미지 **약 1,000장을 직접 수집**하고 Roboflow로 어노테이션해 커스텀 YOLO 모델을 학습했습니다.

| 구분 | 1차 모델 | 2차 모델 (개선) |
| --- | --- | --- |
| 목적 | 악기의 구조적 요소 검출 | 연주 장면의 손 위치·프렛 영역 검출 |
| 클래스 | board, nut, fret, string | board, nut, fret zone, left/right hand |
| 프렛 라벨링 | 개별 프렛 단위 | 프렛 영역 단위 |
| 손 라벨링 | 없음 | 왼손/오른손 전체 영역 |

> 1차 모델에서 현(string) 클래스는 AP 약 65%에 그쳤습니다. 현은 얇고 배경과 구분이 어려워 각도·조명 영향을 크게 받기 때문에, 개별 검출을 고집하는 대신 **손 위치와 프렛 영역을 함께 해석하는 방식**으로 전환했고, 이 판단이 최종 운지 추론에 필요한 정보를 안정적으로 확보하는 전환점이 되었습니다.

## 담당 역할

- **오디오 분석 파트** — 피치·BPM 추정과 음표 구간화 구현. 음정 추정 라이브러리(librosa, CREPE, basic-pitch)를 **비교 실험해 pYIN 기반 방식으로 개선**하고, 비전과 병합 가능한 시간 정보 기반 JSON 구조 설계
- **커스텀 YOLO 데이터셋 구축** — 연주 이미지 약 1,000장 직접 수집·Roboflow 어노테이션, 클래스 구성이 다른 모델 2종 학습·비교
- **오디오·비전 결과 병합** — 음표 타이밍 기준 비전 프레임 매칭, 베이스 튜닝 규칙 기반 줄/프렛 후보 계산 로직 구현
- **성과**: 최종 TAB 생성에 필요한 데이터 흐름을 완성해 **전체 악보 생성 정확도를 59% → 85%로 향상**

## 기술 스택

| 구분 | 기술 |
| --- | --- |
| Language | Python (CLI · JSON 기반 파이프라인) |
| Audio | librosa (pYIN), Demucs, soundfile, SciPy |
| Vision | OpenCV, MediaPipe, Ultralytics YOLO, PyTorch, Roboflow |
| Score Export | music21, MusicXML, ASCII TAB |
| Data / Experiment | pandas, scikit-learn, Pillow, PyYAML |

## 실행 방법

```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt

# 원본 영상에서 전체 분석
python run_full_pipeline.py --video data/videos/test_video6.mp4 --no_display

# 기존 오디오/비전 결과 재사용 (병합~MusicXML만 재실행)
python run_full_pipeline.py --video data/videos/test_video6.mp4 \
  --audio_json outputs/audio/result.json \
  --left_json outputs/vision/left_frames.json \
  --right_json outputs/vision/right_frames.json \
  --skip_audio --skip_vision_detection --no_display
```

## 프로젝트 구조

```text
run_full_pipeline.py     # 전체/부분 파이프라인 실행
run_audio_analysis.py    # 오디오 피치·BPM·음표 분석
src/
  audio/                 # pYIN 피치 분석, 음표 구간화
  detect_left_hand_frets.py / detect_right_hand_strings.py
  merge_hand_vision.py   # 좌/우손 비전 결과 병합
  run_tab_solver.py      # 줄/프렛 추론
  export_musicxml.py     # MusicXML 내보내기
outputs/                 # 실행 결과 (JSON, TAB, MusicXML)
docs/PROJECT_STRUCTURE.md
```

> 대용량 입력 영상·오디오·모델 가중치는 원본 보존을 위해 로컬에만 두고 Git에서는 제외합니다.