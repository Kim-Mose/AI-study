# ==================================================
# 배열 참조
# ==================================================

import numpy as np

arr = np.arange(10)
print(arr) # [0 1 2 3 4 5 6 7 8 9]

arr_sli = arr[0:4]
print(arr_sli) # [0 1 2 3]

# 슬라이싱한 배열 수정
arr_sli[0] = 4
print(arr_sli) # [4 1 2 3]
# 원본 배열도 바뀌어버린다.
print(arr) # [4 1 2 3 4 5 6 7 8 9]

# 이러한 배열 참조를 방지하려면?
# 1. copy() 사용
arr = np.arange(10)

arr_copy = arr.copy()
print(arr_copy)

arr_copy[0] = 4
print(arr_copy) # [4 1 2 3 4 5 6 7 8 9]
print(arr) # [0 1 2 3 4 5 6 7 8 9]

# 2. fancy 인덱싱 사용
idx = [0, 1, 2, 3]
arr_fancy = arr[idx]

arr_fancy[0] = 4
print(arr_fancy) # [4 1 2 3]
print(arr) # [0 1 2 3 4 5 6 7 8 9]