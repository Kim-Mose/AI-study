# AlexNet PyTorch 구현

> **ImageNet Classification with Deep Convolutional Neural Networks** (Krizhevsky et al., 2012)

## 개요
2012년 ImageNet 대회에서 압도적 성능으로 우승하며 딥러닝 부흥기를 연 모델.<br>
GPU와 ReLU, Dropout, 데이터 증강을 적극 활용했다.

## 폴더 구조
```
AlexNet/
├── README.md
├── paper_review.md
├── model.py
├── train.py
├── experiments/
│   ├── original.md
│   └── ablation.md
└── results/
```

## 모델 구조
| 층 | 입력 | 출력 |
| --- | --- | --- |
| Conv1 (11×11, stride 4) | 224×224×3 | 55×55×96 |
| MaxPool | 55×55×96 | 27×27×96 |
| Conv2 (5×5) | 27×27×96 | 27×27×256 |
| MaxPool | 27×27×256 | 13×13×256 |
| Conv3, 4, 5 (3×3) | 13×13×256 | 13×13×256 |
| MaxPool | 13×13×256 | 6×6×256 |
| FC (4096) → FC (4096) → FC (1000) |  |  |

## 실행 방법
```bash
cd Paper_Implementation/AlexNet
python train.py
```

원본은 ImageNet 데이터셋이지만, 본 구현은 **CIFAR-10**을 224×224로 리사이즈해서 사용한다.

## 핵심 contribution
1. **ReLU** 활성화 함수 도입 (학습 속도 6배 향상)
2. **GPU** 활용 (2개의 GTX 580으로 5~6일 학습)
3. **Dropout** (FC 층에 0.5 비율)
4. **데이터 증강** (랜덤 크롭, 좌우 반전, PCA 컬러 변형)
5. **Local Response Normalization** (현재는 잘 안 씀)

## 참고
- [원본 논문 (PDF)](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
