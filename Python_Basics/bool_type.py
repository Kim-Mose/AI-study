# ==================================================
# 불 : 참(True)과 거짓(False)을 나타내는 자료형
# ==================================================

t = True
f = False

print(t, type(t)) # True <class 'bool'>
print(f, type(f)) # False <class 'bool'>

# ==================================================
# 자료형의 참과 거짓
# - 문자열, 리스트 등에서는 값이 있으면 True, 비어있으면 False
# - 숫자는 0이 아니면 True, 0은 False
# ==================================================

print(bool("Python")) # True
print(bool([])) # False
print(bool(123)) # True
print(bool(0)) # False

# ==================================================
# 논리 연산자
# - and : 양 쪽 조건이 모두 True일 때 True
# - or : 양 쪽 조건 중 하나라도 True일 때 True
# - not : True는 False, False는 True로 바꿈
# ==================================================

# and
print(True and True) # True
print(True and False) # False
print(False and False) # False

# or
print(True or True) # True
print(True or False) # True
print(False or False) # False

# not
print(not True) # False
print(not False) # True
