# ==================================================
# 넘파이 배열의 입출력
# ==================================================

import numpy as np

# 배열 저장
arr = np.arange(10)

np.save("my_arr", arr) # .npy 확장자로 저장된다.

# 저장한 배열 불러오기
my_arr = np.load("my_arr.npy")
print(my_arr) # [0 1 2 3 4 5 6 7 8 9]

# 다중 배열 저장하기
arr1 = np.arange(10)
arr2 = np.arange(10, 20)

np.savez("my_arrz", a1 = arr1, a2 = arr2) # key값을 설정하여 저장하고 .npz 확장자로 저장된다.

# 다중 배열 불러오기
my_arrz = np.load("my_arrz.npz")
print(my_arrz) # NpzFile 'my_arrz.npz' with keys: a1, a2 -> 키값으로 저장된 것을 알 수 있다.
print(my_arrz.files) # ['a1', 'a2'] -> 키값이 불러와진다.

print(my_arrz["a1"]) # [0 1 2 3 4 5 6 7 8 9]
print(my_arrz["a2"]) # [10 11 12 13 14 15 16 17 18 19]
