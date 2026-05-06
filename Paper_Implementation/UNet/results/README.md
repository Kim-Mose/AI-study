# UNet 학습 결과

`python train.py` 실행 시 자동으로 생성되는 파일들:

- `training_log.csv` - 에폭별 학습 로그
- `training_curve.png` - 학습 곡선 그래프
- `summary.md` - 학습 요약 (설정 + 최종 성능)
- `*.pth` - 학습된 모델 가중치

## 실행 방법
```bash
cd Paper_Implementation/UNet
python train.py
```
