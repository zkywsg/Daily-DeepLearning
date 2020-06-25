### 1.高阶函数
- 1.1 map
- 1.2 reduce
- 1.3 filter
- 1.4 sorted

```python
# 1.1 map()函数：把一个函数作用到一个Iterable的东西上
# 参数：1.函数 2.Iterable
# 返回值类型：map object
# 举例：把平方作用到列表的每个值中
def f(x):
    return x * x
r = map(f,[1,2,3,4,5,6,7,8,9])
list(r)
```
\>>>[1, 4, 9, 16, 25, 36, 49, 64, 81]

```python
# map()函数实际上就是不是运算规则抽象化了
# 把list的每个数字变成str
list(map(str,[1,2,3,4,5,6,7,8,9]))
```
\>>>['1', '2', '3', '4', '5', '6', '7', '8', '9']

```python
# 1.2 reduce()函数：把一个函数累计作用到序列的下一个元素中
# reduce(f,[x1,x2,x3]) == f(f(x1,x2),x3)
# 举例，把[1,2,3]变成123
from functools import reduce
def fn(x,y):
    return x*10+y
reduce(fn,[1,2,3])
```
\>>> 123

```python
# 1.3 filter函数：接收函数和序列，根据函数作用在序列的每个元素返回的True和False决定是否保留该元素
# 返回类型：Iterator
# 举例：只保留序列中的奇数
def is_odd(n):
    return n%2==1
list(filter(is_odd,[1,3,5,7,2,4,6,8]))
```
\>>>[1, 3, 5, 7]

```python
# 把一个序列中的空字符去掉
def not_empty(s):
    return s.strip()
list(filter(not_empty, ['A', '', 'B', 'C', '  ']))
```
\>>>['A', 'B', 'C']

```python
# sorted对list进行排序
sorted([32,12,-19,55,3])
```
\>>>[-19, 3, 12, 32, 55]

```python
# 参数：key，接收自定义排序函数
# 举例：按照绝对值大小排序
sorted([32,12,-19,55,3],key=abs)
```
\>>>[3, 12, -19, 32, 55]

```python
# 对于字符 是按照ASCII码进行排序
sorted(['sb','Sb','SB'])
```
\>>>['SB', 'Sb', 'sb']

### 2.返回函数
- 2.1 函数作为返回值
- 2.2 闭包

```python
# 2.1 高阶函数出了可以接收函数作为参数，还可以把函数作为结果返回
# 一个可变参数的求和，如果不需要立刻求和，而是在后面的代码中，根据需要再计算
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum
# 调用的时候返回的是求和函数
f = lazy_sum(1,2,3,4)
f

>>> <function __main__.lazy_sum.<locals>.sum()>

# 调用函数f的时候，才是返回真正的求和结果
f()
```
\>>>10

```python
# 在函数lazy_sum里面又定义了函数sum，内部函数可以引用外部函数的参数和局部变量
# lazy_sum返回了sum的时候，相关的参数和变量都保存在了返回函数中，就是闭包
# 每一次调用 返回的都是新的函数 但是传入的是相同的参数
f1 = lazy_sum(1,2,3,4)
f2 = lazy_sum(1,2,3,4)
f1 == f2
```
\>>>False

```python
# 2.2 闭包
# 要注意返回的函数没有立刻执行，要再一次的调用才会执行
def count():
    fs = []
    for i in range(1,4):
        def f():
            return i * i
        fs.append(f)
    return fs
count()
f1,f2,f3 = count()
print(f1(),f2(),f3())
# 因为返回函数引用了i，并且并非立刻执行，3个函数都返回时，所以它引用的变量i已经变成3
```
\>>>9 9 9

```python
# 如果一定要引用循环的变量，就应该再创建一个函数，绑定循环变量
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs = []
    for i in range(1,4):
        fs.append(f(i))
    return fs
count()
f1,f2,f3 = count()
print(f1(),f2(),f3())

```
\>>>1 4 9

### 3.匿名函数
```python
list(map(lambda x: x*x,[1,2,3]))
```
\>>>[1, 4, 9]

```python
# 相当于
def f(x):
    return x * x
```

### 4.装饰器

```python
# 函数也是对象，所以可以赋值给变量，通过变量来调用函数
def now():
    print('2020-01-01')
f = now
f()
```
\>>>2020-01-01

```python
# 想要增强now()的功能，在调用的前打印日志，有不改变函数本身的定义，这样动态增加功能的方法叫做装饰器
# 本质上是一个返回函数的高阶函数
def log(func):
    def wrapper(*args,**kw):
        print('call %s:'%func.__name__)
        return func(*args,**kw)
    return wrapper
# %%
@log
def now(x):
    print('2020-01-01')
now(1)
# %%
```
\>>>
<br/>
call now:
<br/>
2020-01-01

```python
# @log 放在now函数之前 相当于
now = log(now)
```

```python
# decorator也要传入参数，需要编写一个decorator的高阶函数
def log(text):
    def decorator(func):
        def wrapper(*args,**kw):
            print('%s %s:'%(text,func.__name__))
            return func(*args,**kw)
        return wrapper
    return decorator
# %%
@log('execute')
def now():
    print('2020-01-01')
# %%
now()
```
\>>>
<br/>
execute now:
<br/>
2020-01-01

```python
# 相当于
now = log('execute')(now)
# 先执行了log('execute'),返回了decorator函数，再调用返回函数，参数是now，返回的是wrapper
# 也就是原来的now函数变成了现在的wrapper函数了
now.__name__
```
\>>>'wrapper'

```python
# 想要保持now的name的话 wrapper.__name__ = func.__name__
# 可以用装饰器 @functools.wraps(func)
import functools
def log(func):
    @functools.wraps(func)
    def wrapper(*args,**kw):
        print('call %s():' % func.__name__)
        return func(*args,**kw)
    return wrapper

# %%
@log
def now():
    print('2020-01-01')
# %%
now.__name__
```
\>>>'now'
