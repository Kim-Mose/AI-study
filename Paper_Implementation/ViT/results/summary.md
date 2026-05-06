# ViT 학습 결과

- 학습 시작: 2026-05-06 11:14:49
- 학습 종료: 2026-05-06 11:20:08
- 소요 시간: 0:05:19.189571

## 학습 설정

| 항목 | 값 |
| --- | --- |
| model | ViT-Small (CIFAR-10용) |
| dataset | CIFAR-10 |
| image_size | 32 |
| patch_size | 4 |
| embed_dim | 192 |
| depth | 6 |
| heads | 8 |
| batch_size | 64 |
| epochs | 5 |
| learning_rate | 0.0003 |
| optimizer | AdamW |
| weight_decay | 0.05 |
| note | demo run with reduced epochs (원본은 30) |
| device | mps |

## 학습 로그

| epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- |
| 1 | 1.6408 | 0.3932 | 1.3448 | 0.5091 |
| 2 | 1.2551 | 0.5454 | 1.1935 | 0.5652 |
| 3 | 1.0904 | 0.6056 | 1.1121 | 0.5891 |
| 4 | 0.9544 | 0.6550 | 1.0519 | 0.6242 |
| 5 | 0.8314 | 0.7043 | 1.0001 | 0.6449 |

## 최종 성능

- epoch: 5
- train_loss: 0.8314
- train_acc: 0.7043
- test_loss: 1.0001
- test_acc: 0.6449

