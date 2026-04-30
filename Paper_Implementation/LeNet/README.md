# LeNet-5 PyTorch 구현

> **Gradient-Based Learning Applied to Document Recognition** (LeCun et al., 1998)

## 개요
CNN의 시초가 된 모델 LeNet-5를 PyTorch로 구현하고 MNIST 데이터셋으로 학습한 프로젝트.

## 폴더 구조
```
LeNet/
├── README.md           # 이 파일
├── paper_review.md     # 논문 리뷰
├── model.py            # LeNet 모델 정의
├── train.py            # MNIST 학습 코드
├── experiments/
│   ├── original.md     # 논문 결과 재현
│   └── ablation.md     # 추가 실험 (변형/비교)
└── results/            # 학습 결과 그래프, 모델 가중치
```

## 모델 구조
| 층 | 입력 | 출력 | 커널/스트라이드 |
| --- | --- | --- | --- |
| Conv1 + Sigmoid | 28×28×1 | 28×28×6 | 5×5, padding=2 |
| AvgPool1 | 28×28×6 | 14×14×6 | 2×2, stride=2 |
| Conv2 + Sigmoid | 14×14×6 | 10×10×16 | 5×5 |
| AvgPool2 | 10×10×16 | 5×5×16 | 2×2, stride=2 |
| Flatten | 5×5×16 | 400 | - |
| FC1 + Sigmoid | 400 | 120 | - |
| FC2 + Sigmoid | 120 | 84 | - |
| FC3 | 84 | 10 | - |

## 실행 방법

### 환경 설정
```bash
pip install torch torchvision
```

### 학습
```bash
cd Paper_Implementation/LeNet
python train.py
```

### 학습 설정
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch size: 64
- Epochs: 10
- Device: 자동 감지 (CUDA / MPS / CPU)

## 핵심 아이디어
1. **Local Receptive Field** - 합성곱으로 지역적 패턴 학습
2. **Weight Sharing** - 같은 필터를 이미지 전체에 적용
3. **Subsampling (Pooling)** - 위치 변화에 강건한 특징 추출

이 세 가지 원리는 현대 CNN에도 그대로 적용된다.

## 참고 자료
- [원본 논문 (PDF)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [Yann LeCun 홈페이지](http://yann.lecun.com/)
