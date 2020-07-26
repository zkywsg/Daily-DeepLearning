### 问题：把一个序列拆成一个又一个变量
------
### 举个栗子
```python
# 像列表/集合这种特征典型的序列，可以这样具象的赋值给一个个变量
data = ['zkywsg',23,(1997,1,1)]
name,years,birth = data
print name,years,birth
>>>
zkywsg 23 (1997, 1, 1)
```

```python
# 字符串这种可迭代对象也可以拆分
string = 'sb'
a,b = string
print a,b
>>>
s b
```

-----
### 问题：可迭代对象比较长，变量不够的时候，怎么给变量赋值一长段的可迭代对象呢
-----
### 举个例子
```python
# 现在有个人，他有两个电话号码，在一个集合里面，怎么把他们赋值给变量呢？ 用*可以囊括多个迭代对象中的小值
info = ('zkywsg',23,'phone1','phone2')
name,years,*numbers = info
print(name,years,numbers)
>>>
zkywsg 23 ['phone1', 'phone2']
```

```python
# 字符串切割的时候，用一个什么符号做了切割，要最前面，最后面的值。
line = 'nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false'
unname,*field,homedir,sh = line.split(':')
unname
>>>'nobody'
field
>>>['*', '-2', '-2', 'Unprivileged User']
```

```python
# 解压一些元素后又不想要，可以使用*_的组合使用方式
record = ('ACME', 50, 123.45, (12, 18, 2012))
name,*_,(*_,year) = record
name
>>>'ACME'
year
>>>2012
```
-------
### 问题：就是想保存最后几个元素
______
### 举个栗子
```python
# 使用collections.deque可以构造一个固定队列，当队列满了之后，就把老的扔了
from collections import deque
q = deque(maxlen=3)
q.append(1)
q.append(2)
q.append(3)
q
>>>deque([1, 2, 3])
# 再新增元素的时候
q.append(4)
q
>>>deque([2, 3, 4])
```

```python
# 如果不设定长度，那就像正常的队列一样可以操作
q = deque()
q.append(1)
q.append(2)
q.appendleft(3)
q
>>>deque([3, 1, 2])
q.pop()
>>>2
q.popleft()
>>>3
```

--------
### 问题：找一个集合里面最大和最小的N个元素列表
-------
### 举个栗子
```python
# heapq模块有两个函数：nlargest和nsmallest
import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(2,nums))
print(heapq.nsmallest(2,nums))
>>>
[42, 37]
[-4, 1]
```

```python
# 对于这两个函数 可以传入key参数
portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap=heapq.nsmallest(3,portfolio,key=lambda s:s['price'])
cheap
>>>
[{'name': 'YHOO', 'shares': 45, 'price': 16.35},
 {'name': 'FB', 'shares': 200, 'price': 21.09},
 {'name': 'HPQ', 'shares': 35, 'price': 31.75}]
```

```python
# 堆数据结构中heap[0]用还是最小的元素
import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
heap = list(nums)
heapq.heapify(heap)
heap
>>>[-4, 2, 1, 23, 7, 2, 18, 23, 42, 37, 8]
heapq.heappop(heap)
>>>-4
heapq.heappop(heap)
>>>1
```
-----------
### 问题：怎么给队列设定优先级，每次把优先级最高的弹出来
---------
### 举个栗子
```python
# 用heap实现优先级
import heapq

class PriorityQueue():
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self,item,priority):
        # heappush保持优先级最高的在第一个
        heapq.heappush(self._queue,(-priority,self._index,item))
        self._index += 1
    
    def pop(self):
        return heapq.heappop(self._queue)[-1]

class Item():
    def __init__(self,name):
        self.name = name
    def __repr__(self):
        return 'Item({!r})'.format(self.name)

q = PriorityQueue()
q.push(Item('foo'),1)
q.push(Item('bar'), 5)
q.push(Item('spam'), 4)
q.push(Item('grok'), 1)
q.pop()
```



