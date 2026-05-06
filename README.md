# AI Study

> AI/ML 학습 기록 및 논문 구현 저장소

대학원 진학을 목표로 Python 기초부터 딥러닝 논문 구현까지 체계적으로 학습한 기록을 정리한 저장소입니다.

## 목차
- [학습 로드맵](#학습-로드맵)
- [프로젝트 구조](#프로젝트-구조)
- [논문 구현 결과](#논문-구현-결과)
- [환경](#환경)

## 학습 로드맵

```
Python 기초
    ↓
NumPy / Pandas (데이터 처리)
    ↓
선형대수학 (수학 기초)
    ↓
머신러닝 (전통 알고리즘)
    ↓
딥러닝 + PyTorch
    ↓
논문 구현 (LeNet → Transformer → ViT)
```

## 프로젝트 구조

### 1. [Python_Basics](Python_Basics/)
파이썬 기본 문법 정리
- 자료형 (숫자, 문자열, 리스트, 튜플, 딕셔너리, 셋, 불리언)
- 변수, 함수 (`*args`, `**kwargs`, 패킹/언패킹)
- 클래스 (상속, `super`, 클래스/정적 메서드)
- 모듈, 예외 처리, 파일 I/O

### 2. [Numpy](Numpy/)
다차원 배열 라이브러리
- 배열 생성, 인덱싱(Boolean/Fancy), 슬라이싱
- 브로드캐스팅, 축(axis) 연산
- 입출력 (`.npy`, `.npz`)

### 3. [Pandas](Pandas/)
표 형태 데이터 처리
- Series, DataFrame
- `iloc` / `loc` 인덱싱
- 추가/삭제/필터링

### 4. [Linear_Algebra](Linear_Algebra/)
머신러닝의 수학적 기초 (PDF)
- 행렬, 벡터, 가우스 소거법, LU 분해
- 벡터공간, 기저, 차원, 랭크
- 선형독립과 종속

### 5. [Machine_Learning](Machine_Learning/)
전통 머신러닝 알고리즘
- [선형 회귀](Machine_Learning/linear_regression.ipynb) — 경사하강법, MSE
- [로지스틱 회귀](Machine_Learning/logistic_regression.ipynb) — 시그모이드, BCE
- [KNN](Machine_Learning/knn.ipynb)
- [의사결정나무](Machine_Learning/decision_tree.ipynb) — 지니, 엔트로피, 정보 이득
- [랜덤포레스트](Machine_Learning/random_forest.ipynb)
- [SVM](Machine_Learning/svm.ipynb)

### 6. [Deep_Learning](Deep_Learning/)
딥러닝 기초 (PyTorch)
- [딥러닝 개요](Deep_Learning/deep_learning_overview.ipynb)
- [PyTorch 기초](Deep_Learning/pytorch_basics.ipynb)
- [퍼셉트론](Deep_Learning/perceptron.ipynb) → [MLP](Deep_Learning/mlp.ipynb)
- [역전파](Deep_Learning/backpropagation.ipynb)
- [활성화 함수](Deep_Learning/activation_functions.ipynb), [옵티마이저](Deep_Learning/optimizers.ipynb)
- [CNN](Deep_Learning/cnn.ipynb), [RNN/LSTM](Deep_Learning/rnn_lstm.ipynb)
- [어텐션](Deep_Learning/attention.ipynb), [트랜스포머](Deep_Learning/transformer.ipynb)

### 7. [Data_Preprocessing](Data_Preprocessing/)
실무 전처리 기법
- 결측치/이상치 처리
- 스케일링 (StandardScaler, MinMaxScaler, RobustScaler)
- 인코딩 (Label, One-Hot, Ordinal)
- 피처 엔지니어링
- 데이터 분할, 교차 검증
- 불균형 데이터 (SMOTE)

### 8. [Paper_Implementation](Paper_Implementation/) ⭐
주요 논문 9편 PyTorch 구현 + 실험

| 논문 | 연도 | 분야 |
| --- | --- | --- |
| [LeNet](Paper_Implementation/LeNet/) | 1998 | CNN의 시초 |
| [AlexNet](Paper_Implementation/AlexNet/) | 2012 | 딥러닝 부흥 |
| [VGGNet](Paper_Implementation/VGGNet/) | 2014 | 깊이의 가치 |
| [ResNet](Paper_Implementation/ResNet/) | 2015 | 잔차 연결 |
| [UNet](Paper_Implementation/UNet/) | 2015 | 세그멘테이션 |
| [Word2Vec](Paper_Implementation/Word2Vec/) | 2013 | 단어 임베딩 |
| [Transformer](Paper_Implementation/Transformer/) | 2017 | Attention is All You Need |
| [BERT](Paper_Implementation/BERT/) | 2018 | 양방향 사전학습 |
| [ViT](Paper_Implementation/ViT/) | 2020 | Vision Transformer |

각 논문 폴더에는 다음이 포함되어 있습니다:
- `paper_review.md` — 논문 리뷰 (배경, 핵심 아이디어, 결과, 면접 질문)
- `model.py` — PyTorch 구현
- `train.py` — 학습 스크립트
- `experiments/` — 원본 결과 재현 + Ablation 실험
- `results/` — 실제 학습 결과 (CSV, 곡선 그래프, summary)

## 논문 구현 결과

자세한 종합 보고서는 [Paper_Implementation/RESULTS.md](Paper_Implementation/RESULTS.md) 참고.

### 핵심 결과

| 모델 | 데이터셋 | 결과 |
| --- | --- | --- |
| LeNet | MNIST | **Test Acc 98.49%** |
| ResNet18 | CIFAR-10 | Test Acc 76.19% (5 epoch) |
| ViT | CIFAR-10 | Test Acc 64.49% (5 epoch) |
| Transformer | Sequence Reverse | Train Acc 97.57% |
| UNet | 합성 세그멘테이션 | Dice 1.0 |

### 직접 발견한 인사이트
1. **ResNet vs VGG**: 동일 자원에서 ResNet 76% vs VGG 10%(학습 실패) — 잔차 연결과 BatchNorm의 결정적 가치를 실험으로 확인
2. **ViT < ResNet (작은 데이터셋)**: 64.49% vs 76.19% — ViT 논문의 "큰 데이터에서 빛난다"는 주장을 실험으로 검증
3. **Sigmoid 기반 LeNet도 충분**: 현대 활성화 없이도 MNIST에서 98.5% 가능

## 환경

- **OS**: macOS (Apple Silicon)
- **GPU 가속**: MPS (Metal Performance Shaders)
- **Python**: 3.9
- **주요 라이브러리**: PyTorch 2.8, NumPy, Pandas, scikit-learn, Matplotlib

## 사용법

```bash
# 저장소 클론
git clone https://github.com/Kim-Mose/AI-study.git
cd AI-study

# 가상환경 + 의존성
pip install torch torchvision numpy pandas scikit-learn matplotlib

# 논문 구현 학습 예시
cd Paper_Implementation/LeNet
python train.py
# → results/ 에 학습 곡선, 로그, 모델 가중치 자동 저장
```

## 라이선스

[MIT License](LICENSE)
