# ==================================================
# 클래스 : 객체를 만들기 위한 설계도
# ==================================================

# 다음과 같은 계산기 함수가 있다고 하자.

result = 0

def add(num):
    global result
    result += num
    print(result)

add(2) # 2
add(3) # 3

# 만약 계산기가 두 개 필요하면 어떡하지? -> 함수를 두 개 만들어야 되나? -> 클래스 사용

class Add:
    def __init__(self): # 생성자 : 인스턴스가 만들어질 때 자동으로 호출되어 초기값을 설정
        self.result = 0

    def add(self, num): # self : 인스턴스 자기 자신을 가리키는 변수 -> adder.add()로 호출 시 self에는 adder가 들어간다.
        self.result += num
        return self.result
    
# 객체(인스턴스) 생성
# 객체 : 클래스로부터 만들어진 것 자체 -> adder1
# 인스턴스 : 특정 클래스와의 관계를 강조할 때 사용 -> adder1은 Add 클래스의 인스턴스이다.
# 속성 : 클래스 내의 변수 -> self.result 등
# 관점이 다른 것 
adder1 = Add()
adder2 = Add()

adder1_res1 = adder1.add(2) # 2
adder1_res2 = adder1.add(3) # 5
print(adder1_res1)
print(adder1_res2)


adder2_res1 = adder2.add(4) # 4
adder2_res2 = adder2.add(5) # 9
print(adder2_res1)
print(adder2_res2)

# 간단한 사칙연산 계산기 만들기
class Cal:
    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.result = 0

    def add(self):
        return self.first + self.second
    
    def sub(self):
        return self.first - self.second
    
    def mul(self):
        return self.first * self.second
    
    def div(self):
        return self.first / self.second
    
a = Cal(4, 2)

add = a.add()
sub = a.sub()
mul = a.mul()
div = a.div()

print(add, sub, mul, div) # 6 2 8 2.0

# 상속 : 클래스의 기능을 물려받는다.
class MoreCal(Cal): # 클래스명 오른쪽에 부모클래스명을 넣는다.
    def __init__(self, first, second, third): 
        super().__init__(first, second) # 부모클래스의 속성을 그대로 물려받는다.
        self.third = third # 자식클래스에 새로운 속성을 추가한다.

    def pow(self): # 자식클래스에 부모클래스에는 없는 기능을 추가한다.
        return self.first ** self.second
    
    def add(self): # 부모클래스에 있는 메서드의 기능을 추가한다. -> 메서드 오버라이딩
        return self.first + self.second + self.third
    

a = MoreCal(2, 2, 3)
pow = a.pow()
add = a.add()
sub = a.sub()

print(pow, add, sub) # 4 7 0

# 클래스변수 : 모든 인스턴스가 공유하는 변수
class Dog:
    count = 0

    def __init__(self, name):
        self.name = name

a = Dog("리트리버")
b = Dog("포메")

# 특정 객체변수 값의 변화가 다른 객체변수 값에 영향을 끼치지 않는다.
print(a.name) # 리트리버
print(b.name) # 포메

a.name = "불독"

print(a.name) # 불독
print(b.name) # 포메

# 클래스변수는?
print(a.count) # 0
print(b.count) # 0

Dog.count = 1

print(a.count) # 0
print(b.count) # 0

