# ==================================================
# 넘파이 배열 생성
# ==================================================

import numpy as np

# 기본적인 방법
arr = np.array([1, 2, 3])
print(arr, type(arr)) # [1 2 3] <class 'numpy.ndarray'>

# arange : range와 매우 유사
arr = np.arange(1, 10, 0.1)
print(arr, type(arr))
# [1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
#  2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.  4.1 4.2 4.3 4.4 4.5
#  4.6 4.7 4.8 4.9 5.  5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.  6.1 6.2 6.3
#  6.4 6.5 6.6 6.7 6.8 6.9 7.  7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.  8.1
#  8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.  9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9] <class 'numpy.ndarray'>

# linspace : 특정 구간을 특정 개수로 나누기 위함.
arr = np.linspace(1, 100, 10)
print(arr, type(arr)) # [  1.  12.  23.  34.  45.  56.  67.  78.  89. 100.] <class 'numpy.ndarray'>

# zeros : 0으로 채워진 배열 생성
arr = np.zeros(shape = (3, 2))
print(arr, type(arr))
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]] <class 'numpy.ndarray'>

# ones : 1로 채워진 배열 생성
arr = np.ones(shape = (1, 2))
print(arr, type(arr)) # [[1. 1.]] <class 'numpy.ndarray'>

# full : 원하는 값으로 채워진 배열 생성
arr = np.full(shape = (2, 1), fill_value = 3)
print(arr, type(arr))
# [[3]
#  [3]] <class 'numpy.ndarray'>

# eye : 대각 방향이 1로 채워진 단위 행렬 생성
arr = np.eye(3)
print(arr, type(arr))
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]] <class 'numpy.ndarray'>

# random.randn : 표준정규분포에서 랜던 값을 뽑아 배열 생성
arr = np.random.randn(3)
print(arr, type(arr)) # [1.40446345 0.69415222 0.47656816] <class 'numpy.ndarray'>

