# ESP32-CAM Robot SmolVLA Fine-tuning

## 소개
이 프로젝트는 ESP32-CAM 기반 로봇(전면 카메라 1개, IMU 센서, 좌우 모터 제어)용 경량 Vision-Language-Action 모델인 SmolVLA를 파인튜닝하는 코드를 제공합니다. RTX 4090 GPU 환경에서 효율적인 LoRA 기반 파인튜닝을 지원하며, MQTT 및 Node.js 아키텍처와도 원활히 통합됩니다.

## 주요 기능
- SmolVLA 경량 VLA 모델 기반 파인튜닝
- 전면 카메라 영상, IMU 센서 데이터, 자연어 명령 입력 지원
- 좌우 모터 속도 출력 예측
- LoRA로 메모리 및 연산 최적화
- Weights & Biases를 통한 실시간 학습 모니터링
- uv를 이용한 빠른 패키지 설치와 환경 관리
- 데이터셋 전처리 스크립트 포함

## 프로젝트 구조
esp32-robot-vla/
├── pyproject.toml # 프로젝트 및 의존성 설정
├── uv.lock # uv 패키지 잠금 파일
├── src/
│ └── esp32_robot_vla/ # 주요 코드 (모델 설정, 데이터셋, 학습 함수 등)
├── scripts/ # 데이터 준비 및 학습 실행 스크립트
├── config/
│ └── smolvla_config.yaml # 파인튜닝 설정 파일
├── data/ # 학습 데이터셋 (가공된 MQTT 데이터)
└── outputs/ # 학습 결과 및 체크포인트

## 설치 및 실행

### 1. uv 설치 및 프로젝트 초기화
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init esp32-robot-vla
cd esp32-robot-vla

### 2. 의존성 설치
uv sync
uv sync --all-extras

### 3. 데이터 준비
수집한 MQTT 원시 데이터를 전처리하여 학습용 데이터셋 준비:
uv run scripts/prepare_data.py --input ./raw_data --output ./data/robot_dataset --resize 224 224

### 4. 파인튜닝 실행
uv run scripts/train.py --config config/smolvla_config.yaml

## 참고 및 추가 정보
- 학습 로그 및 모델 체크포인트는 `outputs/` 폴더에 저장됩니다.
- Weights & Biases 연동으로 학습 상태 모니터링 가능
- 데이터셋은 LeRobot 형식 기반이며, 커스텀 데이터셋도 지원
- GPU 메모리 부족 시 LoRA 설정을 조정하여 메모리 사용량 감소 가능

## 라이선스
MIT License

## 문의
프로젝트 관련 문의나 개선 제안은 이슈 트래커에 남겨주세요. 
