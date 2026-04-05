# ============================================================
# 예외 처리 : 프로그램 실행 중 발생하는 오류를 처리하는 방법
# ============================================================

# 예외가 발생하는 경우
# print(4 / 0)           # ZeroDivisionError: 0으로 나눌 수 없다.
# print(int("abc"))      # ValueError: 문자열을 숫자로 변환할 수 없다.
# print(a)               # NameError: 정의되지 않은 변수를 사용했다.
# li = [1, 2, 3]
# print(li[5])           # IndexError: 리스트의 범위를 벗어났다.


# ============================================================
# try, except : 예외가 발생하면 except 블록을 실행한다.
# ============================================================

try:
    print(4 / 0)
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.") # 0으로 나눌 수 없습니다.


# 여러 개의 예외 처리
try:
    print(int("abc"))
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except ValueError:
    print("값이 올바르지 않습니다.") # 값이 올바르지 않습니다.


# 모든 예외를 한번에 처리 -> Exception은 모든 예외의 부모 클래스이다.
try:
    print(4 / 0)
except Exception as e:  # e에 에러 메시지가 담긴다.
    print(f"에러 발생: {e}") # 에러 발생: division by zero


# ============================================================
# try, except, else : 예외가 발생하지 않으면 else 블록을 실행한다.
# ============================================================

try:
    result = 4 / 2
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
else:
    print(f"결과: {result}") # 결과: 2.0


# ============================================================
# try, except, finally : 예외 발생 여부와 상관없이 finally 블록은 항상 실행된다.
# ============================================================

try:
    result = 4 / 0
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.") # 0으로 나눌 수 없습니다.
finally:
    print("프로그램 종료") # 프로그램 종료 -> 예외가 발생해도 실행된다.


# ============================================================
# try, except, else, finally 모두 사용
# ============================================================

try:
    result = 4 / 2
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
else:
    print(f"결과: {result}") # 결과: 2.0
finally:
    print("프로그램 종료") # 프로그램 종료


# ============================================================
# raise : 의도적으로 예외를 발생시킨다.
# ============================================================

def check_age(age):
    if age < 0:
        raise ValueError("나이는 0보다 작을 수 없습니다.")
    print(f"나이: {age}")

try:
    check_age(-1)
except ValueError as e:
    print(f"에러 발생: {e}") # 에러 발생: 나이는 0보다 작을 수 없습니다.


# ============================================================
# 사용자 정의 예외 : 직접 예외 클래스를 만들 수 있다.
# ============================================================

class NegativeNumberError(Exception): # Exception 클래스를 상속받는다.
    def __init__(self, value):
        self.value = value
        super().__init__(f"{value}는 음수입니다.")

def check_positive(num):
    if num < 0:
        raise NegativeNumberError(num)
    return num

try:
    check_positive(-5)
except NegativeNumberError as e:
    print(f"에러 발생: {e}") # 에러 발생: -5는 음수입니다.
