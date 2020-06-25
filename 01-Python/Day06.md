### 1.迭代
- 1.1 collections
- 1.2 enumerate

```python
# 1.1 collections
# for循环只能作用在可迭代对象
# 可以用collections来进行判断
from collections import Iterable
# 字符串是否可迭代
isinstance('abc',Iterable)
```
\>>>True

```python
# list是否可迭代
isinstance([1,2,3],Iterable)
```
\>>>True

```python
# 整数是否可迭代
isinstance(123,Iterable)
```
\>>>False

```python
# 1.2 enumerate
# 对list进行索引
for i , value in enumerate(['a','b','c']):
    print(i,value)
```
\>>>
<br/>
0 a
<br/>
1 b
<br/>
2 c

### 2.列表生成式
```python
list(range(1,11))
```
\>>>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

```python
[x * x for x in range(1,11)]
```
\>>>[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

```python
[m + n for m in 'abc' for n in 'xyz']
```
\>>>['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']

```python
d = {'x':'a','y':'b','z':'c'}
[k + '=' + v for k,v in d.items()]
```
\>>>['x=a', 'y=b', 'z=c']

### 3.生成器
- 3.1 简单写法
- 3.2 next
- 3.3 yield
```python
# 如果列表里的元素可以推理出来，从而不用创建完整的list，从而节省空间，称为generator
# 3.1 最简单就是把[]改成()
L = [x * x for x in range(10)]
L
```
\>>>[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

```python
# generator
g = (x * x for x in range(10))
g
```
\>>><generator object <genexpr> at 0x10c4619a8>

```python
# 3.2 next
# generator 保存的是算法 每次用next()调用计算下一个值
next(g)
next(g)
next(g)
```
\>>>
<br/>
0
<br/>
1
<br/>
4

```python
# generator 是可以迭代的 所以可以用for
g = (x * x for x in range(5))
for n in g:
    print(n)
```
\>>>0
1
4
9
16

```python
# 3.3 yield
# 用yield关键字生成generator
# %%
def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield 3
    print('step 3')
    yield 5
o = odd()
next(o)
next(o)
next(o)
# %%
```
\>>>
<br/>
step 1
1
<br/>
step 2
3
<br/>
step 3
5

### 4.迭代器
- 4.1 Iterable
- 4.2 Iterator
- Iterator对象表示是一个数据流，Iterator对象可以被next()函数调用并不断返回下一个数据，直到没有数据的时候抛出StopIteration错误，我们事前不知道他的长度，只有不断调用next()计算它的下一个数据，所以Iterator的计算是惰性的，甚至可以是无限大的长度，但是list却一定是有限的长度。

```python
# 4.1 Iterable
# 可迭代对象可以直接作用在for循环
# 一类是集合数据类型，如list，tuple，dict，set，str
# 另一类是generator
# isinstance()判断对象是否Iterable
from collections import Iterable
# list是否可迭代
isinstance([],Iterable)
# dict是否可迭代
isinstance({},Iterable)
# str是否可迭代
isinstance('a',Iterable)
# generator是否可迭代
isinstance((x for x in range(10)),Iterable)
# 数字是否可迭代
isinstance(123,Iterable)
```
\>>>
<br/>
True
<br/>
True
<br/>
True
<br/>
True
<br/>
False

```python
# 4.2 Iterable和Iterator
# 可以被next()调用的才是Iterator(迭代器)
# 生成器都是Iterator
# list/dict虽然Iterable，但不是Iterator
from collections import Iterator
isinstance((x for x in range(10)),Iterator)
isinstance([],Iterator)
```
\>>>
<br/>
True
<br/>
False

```python
# list/dict/str等Iterable，可以iter()函数变成Iterator
isinstance(iter([]),Iterator)
isinstance(iter('abc'),Iterator)
```
\>>>
<br/>
True
<br/>
True
