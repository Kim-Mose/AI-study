# AlexNet 학습 결과

- 학습 시작: 2026-05-06 11:25:13
- 학습 종료: 2026-05-06 11:26:25
- 소요 시간: 0:01:12.021731

## 학습 설정

| 항목 | 값 |
| --- | --- |
| model | AlexNet |
| dataset | CIFAR-10 (224x224 resize, 5000장 subset) |
| batch_size | 64 |
| epochs | 3 |
| learning_rate | 0.001 |
| optimizer | Adam |
| note | demo run with reduced data and epochs (원본은 ImageNet) |
| device | mps |

## 학습 로그

| epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- |
| 1 | 2.1681 | 0.1792 | 2.1536 | 0.2360 |
| 2 | 1.9777 | 0.2568 | 1.9031 | 0.2940 |
| 3 | 1.8319 | 0.3078 | 1.8762 | 0.3300 |

## 최종 성능

- epoch: 3
- train_loss: 1.8319
- train_acc: 0.3078
- test_loss: 1.8762
- test_acc: 0.3300

