# %% markdown
# ### 1. 调用函数
# - 1.1 单参数函数
# - 1.2 多参数函数
# - 1.3 类型转换函数
# - 1.4 函数名是指向一个函数对象的引用

# %%
# 调用函数
# 知道函数名和参数，就可以调用一个函数
# 1.1 单参数函数
print(abs(100))
print(abs(-20))

# %%
# 1.2 多参数函数 max(x,y,z)比较数值大小
max(1,2,3)

# %%
# 1.3 类型转换函数 int() float() str() bool()

# str转int
a = '123'
print('原类型是：',type(a))
print('转化结果：',int(a))
print('当前类型是:',type(int(a)))
print('----------')
# float转int
b = 12.34
print('原类型是：',type(b))
print('转化结果：',int(b))
print('当前类型是:',type(int(b)))
print('----------')
# str转float
c = '12.34'
print('原类型是：',type(c))
print('转化结果：',float(c))
print('当前类型是:',type(float(c)))
print('----------')
# float转str
d = 1.23
print('原类型是：',type(d))
print('转化结果：',str(d))
print('当前类型是:',type(str(d)))
print('----------')
# int转str
e = 100
print('原类型是：',type(e))
print('转化结果：',str(e))
print('当前类型是:',type(str(e)))
print('----------')
# int转bool
f = 1
print('原类型是：',type(f))
print('转化结果：',bool(f))
print('当前类型是:',type(bool(f)))
print('----------')

# %%
# 1.4 函数名是指向一个函数对象的引用
# 不传参数 a现在变成了函数abs
a = abs
a(-1)

# %% markdown
# ### 2. 定义函数
# - 2.1 定义函数结构
# - 2.2 空函数
# - 2.3 参数检查
# - 2.4 多个返回值

# %%
# 2.1 def + 函数名 + (参数) + :
# 自己写一个绝对值的函数
def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x

# 测试我们写的函数
print(my_abs(-5))

# %%
# 2.2 空函数 什么事情都不做
def nothing():
    pass

# %%
# 2.3 参数检查
# 调用函数的时候，如果参数不对，Python解释器会自动检查，抛出TypeError
my_abs(1,2)
# %%
# 但是如果参数的类型不对，Python的解释器是无法帮我们检查到的
my_abs('a')

# %%
# 可以用isinstance()进行数据类型的检查
def my_abs(x):
    if not isinstance(x,(int,float)):
        raise TypeError('bad operand type')
    if x>= 0:
        return x
    else:
        return -x
print(my_abs('a'))

# %%
# 2.4 返回多个值
# 比如在游戏中，给出坐标，位移和角度，计算新的坐标值
import math
def move(x,y,step,angle=0):
    nx = x + step*math.cos(angle)
    ny = y - step*math.sin(angle)
    return nx,ny
r = move(100,100,60,math.pi/6)
print('位移后','x,y:',r)

# %% markdown
# ### 3.函数的参数

# %%
# 3.1 位置参数
# 对于power()函数，x就是一个位置参数，必须传入有且只有一个参数x
def power(x):
    return x*x
power(5)

# %%
# 改进power函数让他可以计算任意次方，x/n都是位置参数
def power(x,n):
    s = 1
    while n > 0:
        n = n-1
        s = s*x
    return s
print('5的平方：',power(5,2))
print('5的立方：',power(5,3))

# %%
# 3.2 默认参数
# 这时如果对power函数只输入x的值，会报错告诉我们缺少n的值
print(power(5))

# %%
# 加入我们的n有一个默认值的话，在不传入n的时候，就会按照默认设定来
def power(x,n=2):
    s = 1
    while n > 0:
        n = n-1
        s = s*x
    return s
# power(5,2)和power(5)结果是一样的,并且不会报错
print(power(5,2))
print(power(5))

# %%
# 默认参数 -> 使得调用函数的复杂减小
# 录入入学资料的时候，年龄和城市可以设为默认参数
def enroll(name,gender,age = 6,city='BeiJing'):
    print('name:',name)
    print('gender:',gender)
    print('age:',age)
    print('city:',city)
    return '------------'

print(enroll('Sb','F'))
print(enroll('Nt','M',7))

# %%
# 易错点
def add_end(L=[]):
    L.append('END')
    return L
print(add_end())
print(add_end())
print(add_end())

# %%
# 似乎函数记住了END，在这默认参数L指向了[]，他是一个变量，所以默认参数发生了改变
# ！默认参数要用不变对象！
def add_end(L=None):
    if L is None:
        L = []
    L.append('END')
    return L
print(add_end())
print(add_end())

# %%
# 3.3 可变参数
# 传入的参数数目是可变的
# 不用可变参数时，要传入多个数字，需要用list或者tuple传进去
def calc(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print('用list传入:',calc([1,2,3]))
print('用tuple传入:',calc((1,2,3)))

# %%
# 同样的函数用可变参数时
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print('简化了调用过程后：',calc(1,2,3))
print('甚至可以传入0个参数:',calc())

# %%
# 当已经有list或者tuple的时候，*+list(tuple)变成可变参数
nums = [1,2,3]
calc(*nums)

# %%
# 3.4 关键字参数
# 可变参数在函数调用的时候组装成tuple，关键字参数也允许0或多个参数名的参数，但会组装成dict，用**
def person(name,age,**kw):
    print('name:',name,'age:',age,'other:',kw)
    return '----------'
print(person('SB',22))
print(person('SB',22,city='GuangZhou'))
print(person('SB',22,city='GuangZhou',job='Engineer'))

# %%
# 简化调用方法
# 先写一个字典
extra = {'city':'Beijing','job':'Engineer'}
person('Sb',22,**extra)

# %%
# 3.5 命名关键字参数
# 可以限制关键字参数，比如只接收city和job作为关键字参数
# 在*之后的都当作关键字参数
def person(name,age,*,city,job):
    print(name,age,city,job)
    return '----------'

# 必须传入参数名，否则会报错
print(person('SB',22,city='GuangZhou',job='Engineer'))

# %%
# 如果已经有可变参数了，就不需要*了
def person(name,age,*args,city,job):
    print(name,age,args,city,job)
    return '----------'
print(person('SB',22,city='GuangZhou',job='Engineer'))

# %%
