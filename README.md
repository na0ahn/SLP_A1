# KWS (Keyword Spotting) - Google Speech Commands v2

**과제:** 경량 키워드 인식 시스템 구현 (Google Speech Commands Dataset v2 기반)

---

## 프로젝트 개요

이 프로젝트는 Google Speech Commands Dataset v2 (GSCv2)를 사용하여 12-클래스 키워드 인식(KWS) 시스템을 구현합니다.

### 12-클래스 구성
- **10개 목표 단어:** yes, no, up, down, left, right, on, off, stop, go
- **silence:** `_background_noise_` 폴더의 배경 소음을 1초 단위로 분할
- **unknown:** 목표 단어 이외의 모든 음성 명령어 샘플

### 주요 특징
- **입력 특징:** Log-Mel Spectrogram (80 mel bins, 40ms 윈도우, 20ms 홉)
- **모델:** BC-ResNet 기반 경량 CNN (~1.06M 파라미터)
- **증강:** RandomTimeShift + SpecAugment
- **정규화:** Dropout + Weight Decay + Label Smoothing
- **실험 추적:** Weights & Biases (오프라인 모드 지원)

---

## 환경 설정

### 필요 조건
- Python 3.9 이상
- 약 10 GB 디스크 공간 (데이터셋 포함)
- RAM 8 GB 이상 권장

### 패키지 설치

```bash
# 기본 설치
pip install torch torchaudio

# 나머지 의존성
pip install numpy matplotlib seaborn pandas scikit-learn tqdm wandb PyYAML soundfile
```

> **참고:** 이 프로젝트는 오디오 로딩에 `soundfile`을 사용합니다 (FFmpeg 불필요).

---

## 데이터 준비

### 방법 1: 자동 다운로드 스크립트 사용

```bash
python scripts/prepare_data.py
```

이 스크립트는:
1. GSCv2 데이터셋 (~2.3 GB) 다운로드
2. `data/speech_commands_v0.02/` 에 압축 해제
3. 데이터셋 무결성 검증

### 방법 2: 수동 다운로드

```bash
mkdir -p data
wget "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz" \
     -O data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands_v0.02
tar -xzf data/speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02
```

### 데이터 디렉토리 구조

```
data/
  speech_commands_v0.02/
    _background_noise_/  ← silence 클래스 생성에 사용
    yes/
    no/
    up/
    down/
    left/
    right/
    on/
    off/
    stop/
    go/
    ...                  ← unknown 클래스로 사용되는 나머지 단어들
    validation_list.txt  ← 공식 검증 분할 파일
    testing_list.txt     ← 공식 테스트 분할 파일
```

---

## 훈련 실행

### 기본 훈련 (40 에포크)

```bash
python train.py
```

또는 명시적으로 설정 파일 지정:

```bash
python train.py --config configs/final.yaml
```

### 주요 옵션

```bash
# 에포크 수 지정
python train.py --epochs 60

# 모델 선택 (dscnn 또는 bcresnet)
python train.py --model bcresnet

# W&B 온라인 모드 활성화
python train.py --wandb_mode online

# 빠른 정상 동작 확인 (2 에포크)
python train.py --sanity
```

### 훈련 모니터링

W&B 오프라인 모드에서 결과 확인:

```bash
# 로컬 W&B 로그 동기화 (온라인 접속 필요)
wandb sync outputs/wandb/

# 또는 로컬 훈련 기록 확인
cat outputs/logs/training_history.json
```

---

## 평가

```bash
# 최적 체크포인트로 테스트 셋 평가
python train.py --eval_only

# 특정 체크포인트 지정
python train.py --eval_only --checkpoint outputs/checkpoints/best_model.pt
```

---

## 체크포인트 경로

| 파일 | 설명 |
|------|------|
| `outputs/checkpoints/best_model.pt` | 검증 정확도 최고 모델 |
| `outputs/checkpoints/checkpoint_latest.pt` | 최근 에포크 체크포인트 |
| `outputs/checkpoints/checkpoint_epochXXX.pt` | 각 에포크별 체크포인트 |

---

## 리포트 자산 경로

| 경로 | 설명 |
|------|------|
| `report_assets/` | 모든 리포트용 그래프 및 표 |
| `report_assets/combined_training_dashboard.png` | 훈련 대시보드 |
| `report_assets/confusion_matrix.png` | 혼동 행렬 |
| `report_assets/normalized_confusion_matrix.png` | 정규화 혼동 행렬 |
| `report_assets/per_class_accuracy.png` | 클래스별 정확도 |
| `report_assets/feature_examples.png` | Log-Mel 특징 예시 |
| `report_assets/augmentation_examples.png` | 증강 효과 예시 |
| `report_assets/results_table.png` | 결과 요약 표 |
| `report_assets/requirement_checklist.md` | 과제 요구사항 체크리스트 |

---

## W&B 온라인 로깅 활성화

기본적으로 오프라인 모드로 실행됩니다. 온라인 모드 전환:

1. W&B 계정에서 API 키 생성: https://wandb.ai/authorize
2. 환경 변수 설정:
   ```bash
   export WANDB_API_KEY="your_api_key_here"
   ```
3. 훈련 실행 시 온라인 모드 지정:
   ```bash
   python train.py --wandb_mode online
   ```

또는 오프라인 실행 후 동기화:
```bash
wandb sync outputs/wandb/
```

---

## 예상 디렉토리 구조

```
SLP_A1/
  README.md
  train.py                    ← 메인 훈련 스크립트
  pyproject.toml
  configs/
    default.yaml              ← 기본 설정
    final.yaml                ← 최종 훈련 설정
  scripts/
    prepare_data.py           ← 데이터 다운로드/준비
  src/
    data/
      dataset.py              ← GSCv2 데이터셋 클래스
      transforms.py           ← 증강 변환
    features/
      logmel.py               ← Log-Mel 특징 추출
      visualize.py            ← 시각화 유틸리티
    models/
      dscnn.py                ← DS-CNN 모델
      bcresnet.py             ← BC-ResNet 모델
      model.py                ← 모델 팩토리
    engine/
      train.py                ← 훈련 루프
      evaluate.py             ← 평가 루프
      losses.py               ← 손실 함수
      scheduler.py            ← 스케줄러
      metrics.py              ← 메트릭
      utils.py                ← 유틸리티
    tracking/
      wandb_logger.py         ← W&B 로거
    report/
      export_assets.py        ← 리포트 자산 생성
  data/
    speech_commands_v0.02/    ← 데이터셋 (별도 다운로드 필요)
  outputs/
    checkpoints/              ← 모델 체크포인트
    logs/                     ← 훈련 기록
    summaries/                ← 데이터 통계
    wandb/                    ← W&B 로컬 로그
  report_assets/              ← 리포트용 그래프
```

---

## 최종 설정

### 특징 추출 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `sample_rate` | 16,000 Hz | 샘플링 레이트 |
| `duration` | 1.0초 | 입력 길이 (= 16,000 샘플) |
| `n_fft` | 640 | FFT 크기 (= win_length) |
| `win_length` | 640 | 윈도우 길이 (40ms at 16kHz) |
| `hop_length` | 320 | 홉 길이 (20ms at 16kHz) |
| `n_mels` | 80 | Mel bin 수 |
| `f_min` | 20 Hz | 최소 주파수 |
| `f_max` | 8,000 Hz | 최대 주파수 |
| `log_eps` | 1e-6 | 로그 변환 안정화 |
| `normalization` | utterance MVN | 발화 단위 평균-분산 정규화 |
| **출력 형태** | **(80, 49)** | mel bins × 프레임 수 |

> **참고:** n_fft=640 (= win_length)를 사용하여 1초 오디오에서 정확히 49 프레임을 생성합니다.
> n_fft > win_length (예: 1024)를 사용하면 47 프레임이 생성됩니다.

### 모델 파라미터

| 항목 | 값 |
|------|-----|
| 모델 | BC-ResNet (경량 KWS 특화 CNN) |
| 파라미터 수 | ~1,062,988 |
| 최대 허용 파라미터 | 2,500,000 |
| 예산 사용률 | 42.5% |

### 훈련 파라미터

| 파라미터 | 값 |
|----------|-----|
| 옵티마이저 | AdamW |
| 학습률 | 1e-3 |
| Weight Decay | 1e-4 |
| 배치 크기 | 256 |
| 에포크 | 40 |
| 스케줄러 | Cosine + Warmup (5 에포크) |
| Grad Clip | 5.0 |
| Early Stopping | patience=12 |

### 증강 및 정규화

| 유형 | 방법 |
|------|------|
| 증강 1 | RandomTimeShift (±100ms) |
| 증강 2 | SpecAugment (time×2 + freq×2) |
| 정규화 1 | Dropout (p=0.2) |
| 정규화 2 | Weight Decay (1e-4) |
| 정규화 3 | Label Smoothing (0.05) |

---

## 최종 결과

> 아래 결과는 훈련 완료 후 업데이트됩니다.

| 메트릭 | 값 |
|--------|-----|
| 최고 검증 정확도 | TBD (훈련 완료 후 업데이트) |
| **최종 테스트 정확도** | **TBD** |
| 최고 에포크 | TBD |

---

## 폴백 / 편차 사항

1. **오디오 로딩:** torchaudio 2.10+가 TorchCodec (FFmpeg 필요)을 기본 백엔드로 사용하므로, `soundfile` (libsndfile 기반)을 대신 사용합니다. WAV 형식의 GSCv2 파일에 완전히 호환됩니다.

2. **n_fft 설정:** 과제 권장 사항은 n_fft=1024이지만, win_length=640 (40ms)으로 49 프레임을 정확히 생성하기 위해 n_fft=640을 사용합니다. STFT 수식 상 n_fft=1024는 47 프레임을 생성합니다.

3. **W&B 모드:** 기본적으로 오프라인 모드로 실행됩니다. API 키 없이도 모든 메트릭이 로컬에 저장됩니다.

4. **silence 클래스:** `_background_noise_` 폴더의 WAV 파일을 1초 단위로 잘라 생성합니다. 각 훈련 스플릿에 비례적으로 배분됩니다.

5. **unknown 클래스:** 목표 10개 단어 이외의 모든 GSCv2 명령어를 사용하며, 목표 단어 클래스 크기의 약 1.5배로 서브샘플링합니다.

---

## 라이선스

MIT License

---

*이 프로젝트는 SKKU Speech Language Processing 수업 과제 1 (Assignment 1)로 구현되었습니다.*
