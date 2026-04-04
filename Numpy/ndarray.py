# ==================================================
# 넘파이 import와 배열 생성
# ==================================================

import numpy as np # 넘파이는 외부 라이브러리이므로 import 해준다. -> 별칭은 np라 한다.

# 파이썬의 리스트
li = [1, 1.2, "list"]
print(li, type(li)) # [1, 1.2, 'list'] <class 'list'>

# 넘파이의 ndarray
arr = np.array([1, 2, 3])
print(arr, type(arr)) # [1 2 3] <class 'numpy.ndarray'>

# ndarray를 사용하는 이유
# 1. 속도가 빠르다. -> 같은 data type만 저장하므로 연산이 빠르다.
# 2. 벡터 연산이 가능하다.
# 3. 메모리 효율이 좋다.

# ==================================================
# 넘파이 배열의 데이터 특징 알아보기
# ==================================================

arr1 = np.array(0)
arr2 = np.array([1, 2, 3])
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
arr4 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])

# 차원
print(arr1.ndim) # 0 -> 스칼라
print(arr2.ndim) # 1 -> 벡터
print(arr3.ndim) # 2 -> 행렬
print(arr4.ndim) # 3 -> 텐서

# 크기
print(arr1.shape) # ()
print(arr2.shape) # (3,)
print(arr3.shape) # (2, 3)
print(arr4.shape) # (3, 2, 3)

# 데이터 타입
print(arr1.dtype) # int64
print(arr2.dtype) # int64
print(arr3.dtype) # int64
print(arr4.dtype) # int64

arr5 = np.array([1, 2, 3], dtype = np.float64)
print(arr5.dtype) # float64

# 요소 개수
print(arr1.size) # 1
print(arr2.size) # 3
print(arr3.size) # 6
print(arr4.size) # 18
