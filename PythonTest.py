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



