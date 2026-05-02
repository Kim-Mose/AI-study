# U-Net PyTorch 구현

> **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015)

## 개요
의료 영상 세그멘테이션을 위한 모델.<br>
U자 모양의 인코더-디코더 구조와 스킵 연결이 특징이다.

## 폴더 구조
```
UNet/
├── README.md
├── paper_review.md
├── model.py
├── train.py        # 데이터셋 연결 템플릿
├── experiments/
└── results/
```

## 모델 구조
```
입력 ───→ DoubleConv ──────────────────→ DoubleConv ───→ 출력
            ↓ MaxPool                       ↑ ConvTranspose
         DoubleConv ────────────────→ DoubleConv
            ↓ MaxPool                       ↑
         DoubleConv ────────────────→ DoubleConv
            ↓ MaxPool                       ↑
              Bottleneck (DoubleConv)
```
- **인코더**: 다운샘플링하며 특성 추출
- **디코더**: 업샘플링하며 위치 복원
- **스킵 연결**: 인코더의 특성을 디코더에 직접 연결

## 실행 방법
```bash
cd Paper_Implementation/UNet
python model.py  # 모델 구조 확인
```

학습은 세그멘테이션 데이터셋 필요 (예: Carvana, Oxford Pets).

## 핵심 contribution
1. **인코더-디코더 + 스킵 연결**
2. **적은 데이터로 학습 가능** - 의료 영상의 작은 데이터셋에 최적화
3. **세그멘테이션의 표준 모델** - 이후 변형(U-Net++ 등)이 많이 나옴

## 참고
- [원본 논문 (arxiv)](https://arxiv.org/abs/1505.04597)
