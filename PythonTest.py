'''

keys = ['a','b','c','d']
x=dict.fromkeys(keys)
print(x)

y=dict.fromkeys(keys,100)
print(y)

x = {'a':10,'b':20,'c':30,'d':40}
for key in x.keys():
    print(key,end=' ')

keys = ['a','b','c','d']
x={key:value for key, value in dict.fromkeys(keys).items()}
print(x)

fruits = {'strawberry','grape','orange','pineapple','cherry'}
print('orange' in fruits)
print('peach' in fruits)

a=set('apple')
print(a)
b=set(range(5))
print(b)
c={}
print(type(c))
c=set()
print(c)
print(type(c))

a={1,2,3,4}
b={3,4,5,6}
print(a|b)
print(set.union(a,b))

print(a&b)
print(set.intersection(a,b))

print(a-b)
print(set.difference(a,b))

print(a^b)
print(set.symmetric_difference(a,b))

a={1,2,3,4}
a|={5}
print(a)
a={1,2,3,4}
a.update({5})
print(a)

a={1,2,3,4}
a &= {0,1,2,3,4}
print(a)
a={1,2,3,4}
a.intersection_update({0,1,2,3,4})
print(a)

a={1,2,3,4}
a-={3}
print(a)
a={1,2,3,4}
a.difference_update({3})
print(a)

a={1,2,3,4}
a^={3,4,5,6}
print(a)
a={1,2,3,4}
a.symmetric_difference_update({3,4,5,6})
print(a)

a={1,2,3,4}
print(a>={1,2,3,4,})
print(a.issuperset({1,2,3,4}))

a={1,2,3,4}
print(a=={1,2,3,4})
print(a=={4,2,1,3})

a={1,2,3,4}
print(a.isdisjoint({5,6,7,8}))

a={1,2,3,4}
a.add(5)
print(a)
a.remove(3)
print(a)
a.discard(2)
print(a)
a.discard(3)
print(a)

a={1,2,3,4}
print(a.pop())
print(a)
a.clear()
print(a)
a={1,2,3,4}
print(len(a))

a={1,2,3,4}
b=a
b.add(5)
print(a)
print(b)

a={1,2,3,4}
b=a.copy()
b.add(5)
print(a)
print(b)

a={1,2,3,4}
for i in a:
    print(i)

a={i for i in 'apple'}
print(a)
a={i for i in 'pineapple' if i not in 'apl'}
print(a)
    

'''


'''
file=open('hello.txt','w')
file.write('hello.world!!!!!!!!!!!!')
file.colose()



file = open('hello.txt', 'r')
s= file.read()
print(s)
file.close()

with open('hello.txt', 'r') as file:
    s=file.read()
    print(s)

with open('hello.txt','w') as file:
    for i in range(3):
        file.write('Hello,world! {0}\n'.format(i))


lines = ['안녕하세요.\n', '파이썬\n','코딩 도장입니다.\n']
with open('hello.txt','w') as file:
    file.writelines(lines)

with open('hello.txt','r')as file:
    lines = file.readlines()
    print(lines)


with open('hello.txt','r') as file:
    line = None
    while line != '':
        line = file.readline()
        print(line.strip('\n'))


with open('hello.txt','r') as file:
    for line in file:
        print(line.strip('\n'))


'''


'''
피클링,언피클링
'''

'''


import pickle

name = 'james'
age = 17
address = '서울시 서초구 반포동'
scores = {'korean':90,'english':95,'mathematics':85,'science':82}

with open('james.p','wb') as file:
    pickle.dump(name,file)
    pickle.dump(age,file)
    pickle.dump(address,file)
    pickle.dump(scores,file)

import pickle

with open('james.p','rb') as file:
    name = pickle.load(file)
    age = pickle.load(file)
    address = pickle.load(file)
    scores = pickle.load(file)
    print(name)
    print(age)
    print(address)
    print(scores)

    
'''
'''

word = input('단어를 입력하세여: ')
print(word == word[::-1])

word = 'level'
list(word) == list(reversed(word))

'''


'''
text = 'Hello'

for i in range(len(text)-1):
    print(text[i], text[i+1],sep='')

text = 'this is python script'
words = text.split()

for i in range(len(words)-1):
    print(words[i], words[i+1])


text = 'hello'

two_gram = zip(text,text[1:])
for i in two_gram:
    print(i[0],i[1],sep='')

text = 'this is python script'
words = text.split()
list(zip(words, words[1:]))

'''






'''
함수
'''


'''

def hello():
    print('Hello,world!')
hello()

def add(a,b):
    print(a+b)
add(10,20)

def add(a,b):
    return a+b
x = add(10,20)
print(x)

def add_sub(a,b):
    return a+b, a-b
x,y=add_sub(10,20)
print(x)
print(y)


def mul(a,b):
    c=a*b
    return c

def add(a,b):
    c=a+b
    print(c)
    d=mul(a,b)
    print(d)

x=10
y=20
add(x,y)



a=[1,2,3,4,5,6,7,8,9,10]
print(list(map(lambda x: str(x) if x%3 == 0 else x, a)))


def f(x,y):
    return x+y
a= [1,2,3,4,5]
from functools import reduce
print(reduce(f,a))



def calc():
    a=3
    b=5
    def mul_add(x):
        return a*x+b
    return mul_add

c = calc()
print(c(1),c(2),c(3),c(4),c(5))



'''
'''

class person:
    def __init__(self):
        self.hello = '안녕하세요.'

    def greeting(self):
        print(self.hello)

james = person()
james.greeting()

'''


'''
클래스
'''


'''

class person:
    def __init__(self,name,age,address,wallet):
        self.hello = '안녕하세염'
        self.name = name
        self.age = age
        self.address = address
        self.__wallet = wallet
    
    def greeting(self):
        print('{0} 저는 {1}입니다.'.format(self.hello, self.name))

    def pay(self,amount):
        if amount > self.__wallet:
            print('돈이 모자라네...')
            return
        self.__wallet -= amount
        print('이제 {0}원 남았네욤.'.format(self.__wallet))

maria = person('마리아',20,'서울시 서초구 반포동',10000)
maria.greeting()

maria.pay(13000)

print('이름:',maria.name)
print('나이:',maria.age)
print('주소:',maria.address)



'''


'''
클래스 method 사용



class Person:
    def __init__(self):
        self.bag = []

    def put_bag(self, stuff):
        self.bag.append(stuff)


james = Person()
james.put_bag('책')

maria = Person()
maria.put_bag('열쇠')

print(james.bag)
print(maria.bag)


class Knight:
    __item_limit = 10

    def print_item_limit(self):
        print(Knight.__item_limit)


x = Knight()
x.print_item_limit()

print(Knight.__item_limit)


class Calc:
    @staticmethod
    def add(a,b):
        print(a+b)

    @staticmethod
    def mul(a,b):
        print(a*b)
    
Calc.add(10,20)
Calc.mul(10,20)


class Person:
    count = 0

    def __init__(self):
        Person.count += 1

    @classmethod
    def print_count(cls):
        print('{0}명 생성되었습니다.'.format(cls.count))

james = Person()
maria = Person()
tk = Person()
mj = Person()

Person.print_count()


class Person:
    count = 0

    def __init__(self):
        Person.count += 1

    @classmethod
    def print_count(cls):
        print('{0}명 생성되었습니다.'.format(cls.count))

james = Person()
maria = Person()

Person.print_count()


class Person:
    def greeting(self):
        print('안녕하세요')

class Student(Person):
    def study(self):
        print('공부하기')

james = Student()
james.greeting()
james.study()


class Person:
    def greeting(self):
        print('안녕하세요')

class PersonList:
    def __init__(self):
        self.Person_list = []
    
    def append_person(self,Person):
        self.Person_list.append(Person)


class Person:
    def __init__(self):
        print('Person __init__')
        self.hello = '안녕하세요'

class Student(Person):
    def __init__(self):
        print('Student __init__')
        super().__init__()
        self.school = '파이썬 코딩 도장'

james = Student()
print(james.school)
print(james.hello)


class Person:
    def __init__(self):
        print('Person __init__')
        self.hello = '안녕하세요'

class Student(Person):
    pass

james = Student()
print(james.hello)


class A:
    def greeting(self):
        print('A')

class B(A):
    def greeting(self):
        print('B')

class C(A):
    def greeting(self):
        print('C')

class D(B,C):
    pass

x=D()
x.greeting()
print(D.mro())


from abc import *

class StudentBase(metaclass=ABCMeta):
    @abstractmethod
    def study(self):
        pass

    @abstractmethod
    def go_to_school(self):
        pass

class Student(StudentBase):
    def study(self):
        print('공부하기')

    def go_to_school(self):
        print('학교가기')

james = Student()
james.study()
james.go_to_school()



class AdvancedList(list):
    def replace(self, old, new):
        while old in self:
            self[self.index(old)] = new

x = AdvancedList([1,2,3,1,2,3,1,2,3])
x.replace(1,100)
print(x)



import math

class Point2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y

p1 = Point2D(x=30,y=20)
p2 = Point2D(x=60,y=50)

print('p1: {} {}'.format(p1.x,p1.y))
print('p2: {} {}'.format(p2.x,p2.y))

a = p2.x - p1.x
b = p2.y - p1.y


c = math.sqrt(math.pow(a,2) + math.pow(b,2))
print(c)


try:
    x = int(input('나눌 숫자를 입력하세요: '))
    y = 10 / x
    print(y)
except:
    print('예외가 발생했습니다.')



y=[10,20,30]

try:
    index, x = map(int, input('인덱스와 나눌 숫자를 입력하세요: ').split())
    z = y[index] / x
except ZeroDivisionError as e:
    print('숫자를 0으로 나눌 수 없습니다.',e)
except IndexError as e:
    print('잘못된 인덱스입니다.',e)
else:
    print(z)
finally:
    print('코드 실행이 끝났습니다.')
    
    

'''


