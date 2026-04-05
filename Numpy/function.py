# ==================================================
# 넘파이의 유용한 함수들
# ==================================================

import numpy as np

# 1. where
# where(조건, 참일 때의 배열, 거짓일 때의 배열)
arr = np.array([1, 2, 3, 4, 5])

# 조건이 True면 "짝수", False면 "홀수"
result = np.where(arr % 2 == 0, "짝수", "홀수")
print(result)  # ['홀수' '짝수' '홀수' '짝수' '홀수']

# sum
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

print(arr.sum()) # 45 -> 전체 합
print(arr.sum(axis = 0)) # [12 15 18] -> 열 방향(위에서 아래로)의 합
print(arr.sum(axis = 1)) # [ 6 15 24] -> 행 방향(왼쪽에서 오른쪽으로)의 합

# mean
arr = np.arange(10).reshape(2, 5)
print(arr)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]

print(arr.mean()) # 4.5 -> 전체 평균
print(arr.mean(axis = 0)) # [2.5 3.5 4.5 5.5 6.5] -> 열 방향(위에서 아래로)의 평균
print(arr.mean(axis = 1)) # [2. 7.] -> 행 방향(왼쪽에서 오른쪽으로)의 평균

# min, max, var, std
arr = np.arange(10).reshape(2, 5)
print(arr)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]

print(arr.min()) # 0
print(arr.min(axis = 0)) # [0 1 2 3 4]
print(arr.min(axis = 1)) # [0 5]

print(arr.max()) # 9
print(arr.max(axis = 0)) # [5 6 7 8 9]
print(arr.max(axis = 1)) # [4 9]

print(arr.var()) # 8.25
print(arr.var(axis = 0)) # [6.25 6.25 6.25 6.25 6.25]
print(arr.var(axis = 1)) # [2. 2.]

print(arr.std()) # 2.8722813232690143
print(arr.std(axis = 0)) # [2.5 2.5 2.5 2.5 2.5]
print(arr.std(axis = 1)) # [1.41421356 1.41421356]

