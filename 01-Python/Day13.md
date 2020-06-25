### 1.datetime
- 1.1获得当前日期和时间
- 1.2获得指定日期和时间
- 1.3timestamp
- 1.4str转换为datetime
- 1.5本地时间转为UTC时间
- 1.6时区转换

```python
# 1.1获得当前日期和时间
from datetime import datetime
now = datetime.now()
print(now)
print(type(now))
# datetime是模块/datetime模块包含了一个datetime类/
# 通过from datetime import datetime /导入的datetime类
# 仅导入import datetime/必须引用全名datetime.datetime
```
\>>>
<br/>
2020-01-29 19:34:03.067962
<br/>
<class 'datetime.datetime'>

```python
# 1.2获得指定日期和时间
from datetime import datetime
dt = datetime(2020,1,20,12,12)
print(dt)
```
\>>>2020-01-20 12:12:00

```python
# 1.3 datetime转为timestamp
# 我们把1970年1月1日 00:00:00 UTC+00:00时区的时刻称为epoch time，记为0
# 1970年以前的时间timestamp为负数，当前时间就是相对于epoch time的秒数，称为timestamp。

# 相当于
timestamp = 0 = 1970-1-1 00:00:00 UTC+0:00
```

```python
# 北京时间相当于
timestamp = 0 = 1970-1-1 08:00:00 UTC+8:00
```
- 可见timestamp的值与时区毫无关系，因为timestamp一旦确定，其UTC时间就确定了，转换到任意时区的时间也是完全确定的，这就是为什么计算机存储的当前时间是以timestamp表示的，因为全球各地的计算机在任意时刻的timestamp都是完全相同的（假定时间已校准)

```python
# 1.3 timestamp
# timestamp可以直接被转化成标准时间
from datetime import datetime
t = 1429417200.0
print(datetime.fromtimestamp(t)) # 本地时间
print(datetime.utcfromtimestamp(t)) # utc时间

```
\>>>
<br/>
2015-04-19 12:20:00
<br/>
2015-04-19 04:20:00

```python
# 1.4str转换为datetime
# 用户输入的日期和时间是字符串/处理的时间日期/
# datetime.strptime()
from datetime import datetime
cday = datetime.strptime('2020-1-1 19:00:00','%Y-%m-%d %H:%M:%S')
print(cday)
```
\>>>2020-01-01 19:00:00


```python
# 现在我们有datetime对象/把它格式化为字符串显示给用户/需要转换为str/
# strftime()
from datetime import datetime
now = datetime.now()
print(now.strftime('%a, %b %d %H:%M'))
```
\>>>Thu, Jan 30 14:14


```python
# 1.5datetime加减
# 对日期和时间进行加减实际就是把datetime往后或往前计算/得到新的datetime
# 加减可以直接用+-运算符/需要使用timedelta
from datetime import datetime,timedelta
now = datetime.now()
now
>>>datetime.datetime(2020, 1, 30, 14, 19, 8, 974859)

now+timedelta(hours=10)
>>>datetime.datetime(2020, 1, 31, 0, 19, 8, 974859)

now-timedelta(days=1)
>>>datetime.datetime(2020, 1, 29, 14, 19, 8, 974859)

now+timedelta(days=2,hours=12)
>>>datetime.datetime(2020, 2, 2, 2, 19, 8, 974859)
```

```python
# 1.5 本地时间转为UTC时间
# tzinfo
from datetime import datetime,timedelta,timezone
tz_utc_8 = timezone(timedelta(hours=8)) # 创建时区UTC+8:00
now = datetime.now()
now
>>>datetime.datetime(2020, 1, 30, 14, 34, 10, 598601)

dt = now.replace(tzinfo=tz_utc_8) # 强制设置为UTC+8:00
dt
>>>datetime.datetime(2020, 1, 30, 14, 34, 10, 598601, tzinfo=datetime.timezone(datetime.timedelta(seconds=28800)))
```

```python
# 1.6时区转换
# 可以通过utcnow()拿到UTC时间/再转换为任意时区的时间
# 拿到UTC时间/强制设置时区UTC+0:00
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
print(utc_dt)
>>>2020-01-30 06:39:29.334860+00:00

# astimezone 把时区转换为北京时间
bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
print(bj_dt)
>>>2020-01-30 14:39:29.334860+08:00
```

### 2.collections
- 2.1 namedtuple
- 2.2 deque
- 2.3 defaultdict
- 2.4 OrderedDict
- 2.5 ChainMap
- 2.6 counter
```python
# 2.1 namedtuple
# 用来创建一个自定义的tuple对象/规定了tuple元素的个数/可以用属性而不是索引来引用tuple的某个元素
# 具备tuple的不变性/又可以根据属性来引用/方便
from collections import namedtuple
Point = namedtuple('Point',['x','y'])
p = Point(1,2)
p.x
>>>1

p.y
>>>2

isinstance(p,Point)
>>>True

isinstance(p,tuple)
>>>True
```

```python
# 2.2 deque
# deque是为了高效实现插入和删除操作的双向列表/适合于队列和栈
from collections import deque
q = deque(['a','b','c'])
q.append('x')
q.appendleft('y') # 还支持popleft()
q
```
\>>>deque(['y', 'a', 'b', 'c', 'x'])

```python
# 2.3 defaultdict
# 使用dict/加入key不存在就会抛出KeyError/如果希望key不存在时/返回默认值
from collections import defaultdict
dd = defaultdict(lambda:'N/A')
dd['key1'] = 'abc'
dd['key1']
>>>'abc'

dd['key2']
>>>'N/A'
```

```python
# 2.4 OrderedDict
# 要保持key的顺序/可以用OrderedDict
from collections import OrderedDict
d = dict([('a',1),('b',2),('c',3)])
d
>>>{'a': 1, 'c': 3, 'b': 2}

od = OrderedDict([('a',1),('b',2),('c',3)])
od
>>>OrderedDict([('a', 1), ('b', 2), ('c', 3)])
```

```python
# key会按照插入的顺序排列/不是key本身排序
od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3
list(od.keys())
```
\>>>['z', 'y', 'x']

```python
# 可以用来实现一个FIFO的dict/当容量超出限制/先删除最早添加的key
from collections import OrderedDict

class LastUpdatedOrderedDict(OrderedDict):

    def __init__(self,capacity):
        super(LastUpdatedOrderedDict,self).__init__
        self._capacity = capacity

    def __setitem__(self,key,value):
        containsKey = 1 if key in self else 0
        if len(self) - containsKey >= self._capacity:
            last = self.popitem(last=False)
            print('remove:',last)
        if containsKey:
            del self[key]
            print('set:',(key,value))
        else:
            print('add:',(key,value))
        OrderedDict.__setitem__(self,key,value)
```

```python
# 2.5 ChainMap
# 把一组dict穿起来并组成一个逻辑上的dict
# ChainMap本身就是dict/但是查找的时候/会按照顺序在内部的dict依次查找
# 例子：应用程序需要传入参数/参数可以通过命令行传入/可以通过环境变量传入/可以有默认参数
# 通过ChainMap实现参数的优先级查找/即先查找命令行参数/如果没有传入/再查找环境变量/如果没有就是用默认参数

from collections import ChainMap
import os,argparse

# 构造缺省参数
defaults = {
    'color':'red',
    'user':'guest'
}

# 构造命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-u','--user')
parser.add_argument('-c','--color')
namespace = parser.parse_args()
command_line_args = {k:v for k,v in vars(namespace).items() if v}

# 组合成ChainMap
combined = ChainMap(command_line_args,os.environ,defaults)

print('color=%s'%combined['color'])
print('user=%s'%combined['user'])
```

```python
# 没有任何参数/打印处默认参数
$ python3 use_chainmap.py
color=red
user=guest
```

```python
# 传入命令行参数/优先使用命令行参数
$ python3 use_chainmap.py -u sb
color=red
user=sb
```

```python
# 同时传入命令行参数和环境变量/命令行参数的优先级高
$ user=admin color=green python3 use_chainmap.py -u sb
color=green
user=sb
```

```python
# 2.6 counter
from collections import Counter
c = Counter()
for ch in 'programming':
    c[ch] = c[ch] + 1
c
>>>Counter({'p': 1, 'r': 2, 'o': 1, 'g': 2, 'a': 1, 'm': 2, 'i': 1, 'n': 1})

c.update('hello')
c
>>>Counter({'p': 1,
         'r': 2,
         'o': 2,
         'g': 2,
         'a': 1,
         'm': 2,
         'i': 1,
         'n': 1,
         'h': 1,
         'e': 1,
         'l': 2})
```

### 3.struct

- python没有专门处理字节的数据类型/但是b'str'可以表示字节/字节数组=二进制str
- c语言中/可以用struct/union处理字节/以及字节和int/float的转换

```python
# struct模块解决bytes和其他二进制数据类型的转换
# struct的pack函数把任意数据类型变成bytes
import struct
struct.pack('>I',10240099)
```
\>>>b'\x00\x9c@c'

- pack的第一个参数是处理指令/'>'表示字节顺序是big-endian/I表示4字节无符号整数

### 4.itertools

- itertools提供了非常有用的用于操作迭代对象的函数
```python
import itertools
natuals = itertools.count(1)
for n in natuals:
    print(n)
# 会创建一个无限的迭代器/所以会打印出自然数序列/停不下来
```

```python
# cycle()会把一个序列无限重复下去
import itertools
cs = itertools.cycle('ABC')
for c in cs:
    print(c)
# 同样一直重复
```

```python
# repeat() 负责把一个元素无限重复下去/提供第二个参数可以限定重复次数
ns = itertools.repeat('A',2)
for n in ns:
    print(n)
```
\>>>
<br/>
A
<br/>
A

```python
# takewhile()根据条件判断来截取一个有限的序列
natuals = itertools.count(1)
ns = itertools.takewhile(lambda x:x <= 3,natuals)
list(ns)
```
\>>>[1, 2, 3]

```python
# chain()可以把一组迭代对象串联/形成一个更大的迭代器
for c in itertools.chain('ABC','XYZ'):
    print(c)

```
\>>>A B C X Y Z


```python
# groupby()把迭代器中相邻的重复元素挑出来
for key,group in itertools.groupby('AAAABBCAAA'):
    print(key,list(group))
```
\>>>
<br/>
A ['A', 'A', 'A', 'A']
<br/>
B ['B', 'B']
<br/>
C ['C']
<br/>
A ['A', 'A', 'A']
