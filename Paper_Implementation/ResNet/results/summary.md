# ResNet18 학습 결과

- 학습 시작: 2026-05-06 11:07:00
- 학습 종료: 2026-05-06 11:09:43
- 소요 시간: 0:02:43.251999

## 학습 설정

| 항목 | 값 |
| --- | --- |
| model | ResNet18 |
| dataset | CIFAR-10 |
| batch_size | 128 |
| epochs | 5 |
| learning_rate | 0.001 |
| optimizer | Adam |
| scheduler | CosineAnnealingLR |
| note | demo run with reduced epochs |
| device | mps |

## 학습 로그

| epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- |
| 1 | 1.5383 | 0.4386 | 1.3112 | 0.5327 |
| 2 | 1.1559 | 0.5889 | 0.9782 | 0.6578 |
| 3 | 0.9565 | 0.6615 | 0.9105 | 0.6889 |
| 4 | 0.8173 | 0.7109 | 0.7228 | 0.7470 |
| 5 | 0.7031 | 0.7514 | 0.6788 | 0.7619 |

## 최종 성능

- epoch: 5
- train_loss: 0.7031
- train_acc: 0.7514
- test_loss: 0.6788
- test_acc: 0.7619

