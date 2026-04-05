# ==================================================
# 넘파이 배열의 인덱싱
# 리스트와 비슷한 방법으로 가능하므로 자세한 설명은 생략
# ==================================================

import numpy as np

arr1 = np.arange(10)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(arr1) # [0 1 2 3 4 5 6 7 8 9]
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# 1차원 배열 인덱싱
print(arr1[0]) # 0
print(arr1[1]) # 1
print(arr1[-1]) # 9
print(arr1[-2]) # 8

# 2차원 배열 인덱싱
print(arr2[0][1]) # 2
print(arr2[1][1]) # 5
print(arr2[0, 1]) # 2 -> [0][1]과 [0, 1]의 결과는 같다.
print(arr2[1, 1]) # 5

# boolean 인덱싱
arr = np.arange(1, 13).reshape(3, 4)
print(arr)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

bool_arr = arr > 5
print(bool_arr)
# [[False False False False]
#  [False  True  True  True]
#  [ True  True  True  True]]

print(arr[bool_arr]) # [ 6  7  8  9 10 11 12]

# fancy 인덱싱
arr = np.arange(100, 111)
print(arr) # [100 101 102 103 104 105 106 107 108 109 110]

idx = [1, 3, 5, 7, 9]
print(arr[idx]) # [101 103 105 107 109]

