### 1.类和实例
- 1.1 基本定义
- 1.2 创建实例
- 1.3 绑定属性
- 1.4 \__init__
- 1.5 数据封装
```python
# 1.1 基本定义
#  class + 类名 + (object) 即从哪个类继承下来的
class student(object):
    pass
```

```python
# 1.2 创建实例
# 类名+()
bart = student()
# 变量bart指向student类的实例，0x1064b1128是内存地址
bart
student
```
\>>>
<br/>
<__main__.student at 0x1064b1128>
<br>
__main__.student

```python
# 1.3 绑定属性
# 可以自由的给实例变量绑定属性
bart.name = 'sb'
bart.name
```
\>>>'sb'

```python
# 1.4 __init__
# 类就像是模版 当我们想创建实例的时候就把一些属性写进去 可以用__init
class student(object):
    # self 就是实例本身
    def __init__(self,name,score):
        self.name = name
        self.score = score

bart = student('sb',0)
bart.name
bart.score
```
\>>> 'sb' 0

```python
# 1.5 数据封装
# 比如一个函数本来就要用到学生类里面的数据，那当然就把函数放在类里面多好嘛
class student(object):
    def __init__(self,name,score):
        self.name = name
        self.score = score
    def print_score(self):
        print('%s:%s'%(self.name,self.score))

bart = student('sb',0)
bart.print_score()
```
\>>>sb:0

### 2.限制访问
```python
# 按照上述的定义，外部代码还可以自由修改实例的属性
bart.score
bart.score = 100
bart.score
```
\>>>
<br/>
0
<br/>
100

```python
# 我们更希望这些属性是私有的，不能被外部代码修改的
class student(object):
    def __init__(self,name,score):
        self.__name = name
        self.__score = score

    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))

bart = student('sb',0)
bart.__name
>>>
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-24-1e120ece361e> in <module>
----> 1 bart.__name

AttributeError: 'student' object has no attribute '__name'
```

```python
# 这么做又有一个问题，虽然外部代码不能修改实例的属性了。但是我们还是希望外部代码可以获得他的值
class student(object):
    def __init__(self,name,score):
        self.__name = name
        self.__score = score

    # 让外部可以获得属性的值
    def get_name(self):
        return self.__name

    def get_score(self):
        return self.__score

    # 让外部可以改变属性的值
    def set_name(self,name):
        self.__name = name

    # 可以通过这样改变属性的方法来做参数检查，避免传入无效参数
    def set_score(self,score):
        if 0 <= score < 100:
            self.__score = score
        else:
            raise ValueError('bad score')

bart = student('sb',0)
bart.get_name()
bart.set_name('SB')
bart.get_name()
```
\>>>
<br/>
'sb'
<br/>
'SB'

```python
bart.set_score(50)
bart.get_score()
>>>50

bart.set_score(250)
>>>
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-40-6cf87971f6eb> in <module>
----> 1 bart.set_score(250)

<ipython-input-31-e38b3fe7e472> in set_score(self, score)
     20             self.__score = score
     21         else:
---> 22             raise ValueError('bad score')

ValueError: bad score
```

### 3. 继承和多态
- 3.1 继承
- 3.2 子类的特性
- 3.3 理解多态
```python
# 现在我们已经有一个动物class，一个run()方法打印
class Animal(object):
    def run(self):
        print("Animal is running...")
```

```python
# 3.1 继承
# 当我们需要具体的动物类时，可以从Animal继承
class cat(Animal):
    pass

class Dog(Animal):
    pass
```

```python
# 继承最大的好处就是子类有了父类的全部功能
dog = Dog()
dog.run()
```
\>>>Animal is running...

```python
# 3.2 子类的特性
# 子类可以增加新的方法也可以对父类的方法进行改进
class Dog(Animal):
    # 改进父类方法
    def run(self):
        print('Dog is running...')

    # 新的方法
    def eat(self):
        print('Dog is eating...')

dog = Dog()
dog.run()
dog.eat()
```
\>>>
<br/>
Dog is running...
<br/>
Dog is eating...

```python
# 3.3 理解多态
# 定义一个class，实际上就是定义了一种数据类型
a = list() # a是list类型
b = Animal() # b是Animal类型
c = Dog() # c是Dog类型
```

```python
# c既是Dog也是Animal
# 就是说Dog可以看成是一个Animal
isinstance(c,Animal)
isinstance(c,Dog)
```
\>>>
<br/>
True
<br/>
True

```python
# 理解多态的好处
class Animal(object):
    def run(self):
        print("Animal is running...")
    def eat(self):
        print('Anumal is eating...')

class Dog(Animal):
    def run(self):
        print('Dog is running...')
    def eat(self):
        print('Dog is eating...')


def run_eat(a):
    a.run()
    a.eat()
run_eat(Animal())
```
\>>>
<br/>
Animal is running...
<br/>
Anumal is eating...

```python
run_eat(Dog())
# 多态的好处：传入的只要是Animal或者他的子类，就会自动调用实际类型的run()
# 调用的时候只管调用，新加一个子类的时候只要保证他继承的方法没写错
# 开闭原则：
# 对扩展开放：允许增加子类
# 对修改封闭：不需要修改类的run_eat()函数
```
\>>>
<br/>
Dog is running...
<br/>
Dog is eating...

- 静态语言：利用多态特性的时候，传入的对象必须严格的是Animal类或者他的子类
- 动态语言：不要求严格的继承体系
  - 鸭子类型：一个对象只要看起来像鸭子，走路也像鸭子，就能被看作是鸭子
  - python：“file-like object”就是一种鸭子类型，某个对象有当前的这个函数方法，就可以当作是这个函数的对象了。

### 4.实例属性和类属性

```python
# 给实例绑定属性
# 1.通过实例变量 2.通过self变量
class student(object):
    def __init__(self,name):
        self.name = name

s = student('sb')
s.score = 0
```

```python
# 直接给类绑定一个类属性
class student(object):
    name = 'student'

# 创建实例
s = student()
# 这个时候实例没有name，所以会向上找类的name
s.name
```
\>>>'student'

```python
# 打印类的属性
print(student.name)
>>>student

# 给实例绑定属性
s.name = 'sb'
print(s.name)
>>>sb

print(student.name)
>>>student
```
