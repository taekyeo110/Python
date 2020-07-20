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

'''

