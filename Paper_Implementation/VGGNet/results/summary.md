# VGG16 학습 결과

- 학습 시작: 2026-05-06 11:31:09
- 학습 종료: 2026-05-06 11:39:52
- 소요 시간: 0:08:43.030331

## 학습 설정

| 항목 | 값 |
| --- | --- |
| model | VGG16 |
| dataset | CIFAR-10 (224x224 resize, 5000장 subset) |
| batch_size | 32 |
| epochs | 3 |
| learning_rate | 0.001 |
| optimizer | Adam |
| note | demo run with reduced data and epochs |
| device | mps |

## 학습 로그

| epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- |
| 1 | 2.3302 | 0.1062 | 2.3227 | 0.1000 |
| 2 | 2.3046 | 0.0970 | 2.3030 | 0.1000 |
| 3 | 2.3033 | 0.1028 | 2.3025 | 0.1000 |

## 최종 성능

- epoch: 3
- train_loss: 2.3033
- train_acc: 0.1028
- test_loss: 2.3025
- test_acc: 0.1000

