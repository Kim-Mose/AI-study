# ============================================================
# 사용자 입출력
# 사용자에게 입력을 받고 출력하는 것을 말한다.
# ============================================================

# 사용자로부터 입력 받기
a = input("보여질 문구") # input함수 안에 사용자로부터 입력 받을 시 보여질 문구를 작성한다.(나이를 입력하시오 등등)
print(a) 

# 저장된 입력은 무슨 data type? -> 문자열로 저장되므로 숫자를 받을 시에 형변환이 필요하다.
num = input("숫자를 입력해주세요 : ")
print(num, type(num)) # 20 <class 'str'>

# int, float를 활용하여 형변환한다.
num = int(input("숫자를 입력해주세요 : "))
print(num, type(num)) # 20 <class 'int'>

# ============================================================
# print 함수 자세히 알기
# ============================================================

# 1. 따옴표와 (+)연산
# 따옴표를 연속적으로 쓰거나 (+)연산을 한다면 띄어쓰기가 되지 않는다.
print("I" "love" "Python") # IlovePython

print("I"+"love"+"Python") # IlovePython

# 띄어쓰기를 하려면 ,를 사용한다.
print("I", "love", "Python") # I love Python


# 2. sep과 end
# sep은 구분자를 선택(기본값 : 공백), end는 마지막 출력 뒤에 실행되는 것(기본값 : 줄바꿈)

# 1. sep 사용하기
print("I", "love", "Python") # I love Python -> 기본적으로 ,가 나오면 띄어쓰기가 된다.

print("I", "love", "python", sep = '\t') # I       love    python -> 구분자를 tab으로 설정하니 ,마다 tab이 된다.


# end 사용하기
# 기본적으로 print 함수를 두 번 사용하면 자동적으로 줄바꿈이 된다.
print("I", "love") # I love
print("python") # python

# 첫번째 print함수의 마지막 출력 뒤에 공백을 출력하게 하니 줄바꿈 없이 문장이 한 줄로 출력된다.
print("I", "love", end = ' ')
print("python") # I love python