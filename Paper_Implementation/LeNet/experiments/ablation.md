# Ablation 실험 (변형 실험)

## 목표
LeNet-5의 각 구성 요소를 변경하면 성능이 어떻게 달라지는지 실험한다.<br>
이를 통해 각 컴포넌트의 역할과 효과를 정량적으로 이해한다.

## 실험 1: 활성화 함수 비교 (Sigmoid vs ReLU)

### 가설
ReLU는 기울기 소실 문제를 완화하므로 Sigmoid보다 학습이 빠르고 성능이 좋을 것이다.

### 설정
| 변수 | 값 |
| --- | --- |
| 활성화 함수 | Sigmoid (원본) vs ReLU |
| 나머지 | 모두 동일 |

### 결과
| 활성화 함수 | Test Acc | 수렴 속도 |
| --- | --- | --- |
| Sigmoid (원본) |  |  |
| ReLU |  |  |

### 분석


## 실험 2: 풀링 방식 비교 (AvgPool vs MaxPool)

### 가설
MaxPool은 가장 두드러진 특성만 살리므로 AvgPool보다 분류 성능이 좋을 것이다.

### 설정
| 변수 | 값 |
| --- | --- |
| 풀링 | AvgPool (원본) vs MaxPool |

### 결과
| 풀링 방식 | Test Acc |
| --- | --- |
| AvgPool (원본) |  |
| MaxPool |  |

### 분석


## 실험 3: BatchNorm 추가

### 가설
BatchNorm은 학습을 안정화하고 더 큰 학습률을 사용할 수 있게 해준다.

### 설정
| 변수 | 값 |
| --- | --- |
| BatchNorm | 없음 (원본) vs Conv 뒤에 추가 |

### 결과
| 모델 | Test Acc | 수렴 속도 |
| --- | --- | --- |
| 원본 |  |  |
| BN 추가 |  |  |

### 분석


## 실험 4: Optimizer 비교 (SGD vs Adam)

### 가설
Adam은 학습률을 자동 조정하므로 SGD보다 빠르게 수렴할 것이다.

### 설정
| 변수 | 값 |
| --- | --- |
| Optimizer | SGD (lr=0.01) vs Adam (lr=0.001) |

### 결과
| Optimizer | Test Acc | 수렴 epoch |
| --- | --- | --- |
| SGD |  |  |
| Adam |  |  |

### 분석


## 실험 5: 데이터셋 변경 (MNIST → Fashion-MNIST)

### 가설
Fashion-MNIST는 MNIST보다 복잡한 패턴이 많아 LeNet의 성능이 떨어질 것이다.

### 설정
| 변수 | 값 |
| --- | --- |
| 데이터셋 | MNIST vs Fashion-MNIST |

### 결과
| 데이터셋 | Test Acc |
| --- | --- |
| MNIST |  |
| Fashion-MNIST |  |

### 분석


## 종합 결론

### 핵심 발견
1. (실험 결과 정리)
2. 
3. 

### 가장 효과적인 개선
- 

### 한계
- LeNet은 단순한 모델이라 작은 변경으로도 큰 차이 보일 수 있음
- 더 복잡한 데이터셋(CIFAR-10 등)에서는 다른 결과 나올 가능성

### 다음 단계
- AlexNet에서 더 큰 데이터셋과 더 깊은 모델로 어떤 변화가 있는지 확인
