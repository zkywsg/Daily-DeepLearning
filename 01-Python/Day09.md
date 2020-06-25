### 1.使用__slots__

```python
class student(object):
    pass

# 给实例绑定一个属性
s = student()
s.name = 'sb'
s.name
```
\>>>'sb'

```python
# 给实例绑定方法
def set_age(self,age):
    self.age = age

from types import MethodType
# 给实例绑定方法
s.set_age = MethodType(set_age,s)
# 调用实例方法
s.set_age(22)
s.age
```
\>>>22

```python
# 对于别的实例无效
s2 = student()
s2.set_age()
>>>
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-11-537c96074839> in <module>
----> 1 s2.set_age()

AttributeError: 'student' object has no attribute 'set_age'
```

```python
# 直接给class绑定方法 所有实例都可以用
def set_score(self,score):
    self.score = score
student.set_score = set_score

s.set_score(0)
s.score

s2.set_score(10)
s2.score
```

```python
# 现在我们想限制实例的属性 比如只给student的实例name和age属性
# 使用__slots__ / 对继承的子类无效
class student(object):
    # 用tuple定义允许绑定的属性
    __slots__ = ('name','age')

s = student()
s.name = 'sb'
s.age = 0
# 绑定其他属性
s.score = 0
>>>
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-22-f3e0e0dee7b4> in <module>
----> 1 s.score = 0

AttributeError: 'student' object has no attribute 'score'
```

### 2. @property

```python
# 绑定属性的时候，虽然方便，但是直接暴露出来，没法检查参数还可以随便改
# 通过set_score,get_score可以检查参数
class student(object):

    def get_score(self):
        return self.__score
    def set_score(self,value):
        if not isinstance(value,int):
            raise ValueError('score must be integer!')
        if value >100 or value < 0:
            raise ValueError('score must between 0-100')
        self.__score = value
s = student()
s.set_score(10)
s.get_score()
```
\>>>10

```python
# 现在觉得太麻烦了 可以用装饰器给函数动态加上功能 @property负责把一个方法变成属性调用
class student(object):
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value

s = student()
s.score = 60 # 转化为了s.set_score(60)
s.score # 转化为了s.get_score
```
\>>>60

### 3.多重继承

```python
# 子类通过继承多种父类，同时获得多个父类的功能
class Animal(object):
    pass

class Mammal(Animal):
    pass

class Bird(Animal):
    pass

class Dog(Mammal):
    pass

class Bat(Mammal):
    pass

class Parrot(Bird):
    pass

class Ostrich(Bird):
    pass
```

```python
# 给动物加上功能，可以通过继承功能类
class Runnable(object):
    def run(self):
        print('running...')

class Flyable(object):
    def fly(self):
        print('flying...')

class Dog(Mammal,Runnable):
    pass

class Bat(Mammal,Flyable):
    pass

# 通过这样多重继承的方式，一个子类可以获得多个父类的功能
```

```python
# MixIn
# 为了更好看出继承关系，Runnable和Flyable改为RunnableMixIn和FlyableMixIn
class Dog(Mammal,RunnableMixIn):
    pass
```

### 4.定制类
- 4.1 \__str__
- 4.2 \__iter__
- 4.3 \__getitem__
- 4.4 \__getattr__
- 4.5 \__call__
```python
# 4.1 __str__
# 先打印一个实例
class student(object):
    def __init__(self,name):
        self.name = name

print(student('sb'))
# 打印出来不好看
>>>
<__main__.student object at 0x1077bba20>


```python
# __str__()方法，可以打印除比较好看的自定义的字符串
class student(object):
    def __init__(self,name):
        self.name = name
    def __str__(self):
        return 'student object (name: %s)' % self.name

print(student('sb'))
```
\>>>student object (name: sb)

```python
# 不用print的时候，打印出来的实例还是不好看
s = student('sb')
s
```
\>>><__main__.student at 0x1078d6630>

```python
# 因为直接显示变量调用的不是__str__(),而是__repr__
class student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'student object (name=%s)' % self.name
    __repr__ = __str__

s = student('sb')
s
```
\>>>student object (name=sb)

```python
# __iter__
# 想要被for循环迭代，这个类就要有__iter__(),返回一个迭代对象，for就会不断调用该调用这个迭代对象的__next__()方法拿到下一个值，直到StopIteration退出循环
class Fib(object):
    def __init__(self):
        self.a,self.b = 0,1
    def __iter__(self):
        return self
    def __next__(self):
        self.a,self.b = self.b,self.a + self.b
        if self.a > 5:
            raise StopIteration
        return self.a

for n in Fib():
    print(n)
```
\>>>1
1
2
3
5

```python
# __getitem__
# 虽然可以用for循环了，但是想要像list一样切片
class Fib(object):
    def __getitem__(self, n):
        if isinstance(n, int): # n是索引
            a, b = 1, 1
            for x in range(n):
                a, b = b, a + b
            return a
        if isinstance(n, slice): # n是切片
            start = n.start
            stop = n.stop
            if start is None:
                start = 0
            a, b = 1, 1
            L = []
            for x in range(stop):
                if x >= start:
                    L.append(a)
                a, b = b, a + b
            return L
f = Fib()
f[0]
f[1:3]
```
\>>>
<br/>
1
<br/>
[1, 2]

```python
# 4.4 __getattr__
# 当没有定义某些属性的时候，通过__getattr__，动态返回一个属性
class student(object):
    def __getattr__(self,attr):
        if attr == 'age':
            return lambda:22
        raise AttributeError('\'Student\' object has no attribute \'%s\'' % attr)

s = student()
s.age()
```
\>>> 22

```python
# 4.5 __call__ 对实例本身进行调用
class student(object):
    def __init__(self,name):
        self.name = name
    def __call__(self):
        print('Your name is %s'%self.name)

s = student('sb')
s()
```
\>>>Your name is sb

```python
# callable()函数 判断一个对象是否可以被调用
callable(student('sb'))
>>>True
callable(max)
>>>True
callable(list)
>>>True
callable('str')
>>>False
```

### 5.枚举类

```python
# enum实现枚举
from enum import Enum
Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

for name,member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)
```
<br/>
Jan => Month.Jan , 1
<br/>
Feb => Month.Feb , 2
<br/>
Mar => Month.Mar , 3
<br/>
Apr => Month.Apr , 4
<br/>
May => Month.May , 5
<br/>
Jun => Month.Jun , 6
<br/>
Jul => Month.Jul , 7
<br/>
Aug => Month.Aug , 8
<br/>
Sep => Month.Sep , 9
<br/>
Oct => Month.Oct , 10
<br/>
Nov => Month.Nov , 11
<br/>
Dec => Month.Dec , 12

```python
# 想要更精确的控制枚举
# @unique 保障没有重复值
from enum import Enum, unique
@unique
class Weekday(Enum):
    Sun = 0 # Sun的value被设定为0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6

day1 = Weekday.Mon
print(day1)
```
\>>>Weekday.Mon
