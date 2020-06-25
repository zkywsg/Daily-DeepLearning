# pandas

### Series
- 类似于一维数组的对象
```python
import pandas as pd
obj = pd.Series([4,7,-5,3])
obj
>>>
0    4
1    7
2   -5
3    3
dtype: int64

obj.values
>>>array([ 4,  7, -5,  3])

obj.index
>>>RangeIndex(start=0, stop=4, step=1)
```

```python
obj2 = pd.Series([4,7,-5,3],index=['d','b','a','c'])
obj2
>>>
d    4
b    7
a   -5
c    3
dtype: int64

obj2.index
>>>Index(['d', 'b', 'a', 'c'], dtype='object')

obj2['a']
>>>-5

obj2['d'] = 6
obj2[['c','a','d']]
>>>
c    3
a   -5
d    6
dtype: int64
```
```python
obj2
>>>
d    6
b    7
a   -5
c    3
dtype: int64

obj2[obj2>0]
>>>
d    6
b    7
c    3
dtype: int64

obj2 * 2
>>>
d    12
b    14
a   -10
c     6
dtype: int64

import numpy as np
np.exp(obj2)
>>>
d     403.428793
b    1096.633158
a       0.006738
c      20.085537
dtype: float64

# 可以把Series看成一个定长的有序字典
'b' in obj2
>>>True

'e' in obj2
>>>False

sdata = {'Ohio':35000,'Texas':70000,'Oregon':34234,'Utah':5000}
obj3 = pd.Series(sdata)
obj3
>>>
Ohio      35000
Texas     70000
Oregon    34234
Utah       5000
dtype: int64

states = ['California','Ohio','Oregon','Texas']
obj4 = pd.Series(sdata,index=states)
obj4
>>>California        NaN
Ohio          35000.0
Oregon        34234.0
Texas         70000.0
dtype: float64

pd.isnull(obj4)
>>>
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool

pd.notnull(obj4)
>>>
California    False
Ohio           True
Oregon         True
Texas          True
dtype: bool

obj4.isnull()
>>>
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool

obj3 + obj4
>>>
California         NaN
Ohio           70000.0
Oregon         68468.0
Texas         140000.0
Utah               NaN
dtype: float64

obj4.name = 'population'
obj4.index.name = 'state'
obj4
>>>
state
California        NaN
Ohio          35000.0
Oregon        34234.0
Texas         70000.0
Name: population, dtype: float64
```

### DataFrame
- 表格型的数据结构
- 虽然是以二维结构保存数据的/但可以轻松表示更高维的数据

```python
import pandas as pd
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
        'year':[2000,2001,2002,2001,2002],
        'pop':[1.5,1.7,3.6,2.4,2.9]}
frame = pd.DataFrame(data)

frame
>>>
state	year	pop
0	Ohio	2000	1.5
1	Ohio	2001	1.7
2	Ohio	2002	3.6
3	Nevada	2001	2.4
4	Nevada	2002	2.9

pd.DataFrame(data,columns=['year','state','pop'])
>>>
year	state	pop
0	2000	Ohio	1.5
1	2001	Ohio	1.7
2	2002	Ohio	3.6
3	2001	Nevada	2.4
4	2002	Nevada	2.9

frame2 = pd.DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five'])
frame2
>>>
year	state	pop	debt
one	2000	Ohio	1.5	NaN
two	2001	Ohio	1.7	NaN
three	2002	Ohio	3.6	NaN
four	2001	Nevada	2.4	NaN
five	2002	Nevada	2.9	NaN

frame2.columns
>>>
Index(['year', 'state', 'pop', 'debt'], dtype='object')

frame2['state']
>>>
one        Ohio
two        Ohio
three      Ohio
four     Nevada
five     Nevada
Name: state, dtype: object

frame2.year
>>>
one      2000
two      2001
three    2002
four     2001
five     2002
Name: year, dtype: int64

frame2.loc['three']
>>>
year     2002
state    Ohio
pop       3.6
debt      NaN
Name: three, dtype: object

frame2['debt'] = 16.5
frame2
>>>
year	state	pop	debt
one	2000	Ohio	1.5	16.5
two	2001	Ohio	1.7	16.5
three	2002	Ohio	3.6	16.5
four	2001	Nevada	2.4	16.5
five	2002	Nevada	2.9	16.5


import numpy as np
frame2['debt'] = np.arange(5.)
frame2
>>>
year	state	pop	debt
one	2000	Ohio	1.5	0.0
two	2001	Ohio	1.7	1.0
three	2002	Ohio	3.6	2.0
four	2001	Nevada	2.4	3.0
five	2002	Nevada	2.9	4.0

val = pd.Series([-1.2,-1.5,-1.7],index=['two','four','five'])
frame2['debt'] = val
frame2

frame2['eastern'] = frame2.state == 'Ohio'
frame2
>>>
year	state	pop	debt	eastern
one	2000	Ohio	1.5	NaN	True
two	2001	Ohio	1.7	-1.2	True
three	2002	Ohio	3.6	NaN	True
four	2001	Nevada	2.4	-1.5	False
five	2002	Nevada	2.9	-1.7	False

del frame2['eastern']
frame2.columns
>>>
Index(['year', 'state', 'pop', 'debt'], dtype='object')
```

```python
import pandas as pd
pop = {'Nevada':{2001:2.4,2002:2.9},
        'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3 = pd.DataFrame(pop)
frame3
>>>
Nevada	Ohio
2001	2.4	1.7
2002	2.9	3.6
2000	NaN	1.5

frame3.T
>>>
2001	2002	2000
Nevada	2.4	2.9	NaN
Ohio	1.7	3.6	1.5

pd.DataFrame(pop,index=[2001,2002,2003])
>>>
Nevada	Ohio
2001	2.4	1.7
2002	2.9	3.6
2003	NaN	NaN

pdata = {'Ohio':frame3['Ohio'][:-1],
        'Nevada':frame3['Nevada'][:2]}
pd.DataFrame(pdata)
>>>
Ohio	Nevada
2001	1.7	2.4
2002	3.6	2.9

frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3
>>>
state	Nevada	Ohio
year
2001	2.4	1.7
2002	2.9	3.6
2000	NaN	1.5

frame3.values
>>>
array([[2.4, 1.7],
       [2.9, 3.6],
       [nan, 1.5]])
```

### 索引对象
```python
import pandas as pd
obj = pd.Series(range(3),index=['a','b','c'])
index = obj.index
index
>>>
Index(['a', 'b', 'c'], dtype='object')

index[1:]
>>>Index(['b', 'c'], dtype='object')
```
- index对象不可修改 才能让index对象在多个数据结构之间安全共享
```python
import numpy as np
index = pd.Index(np.arange(3))
obj2 = pd.Series([1.5,-2.5,0],index = index)
obj2.index is index
>>>True
```
```python
frame3
>>>
state	Nevada	Ohio
year
2001	2.4	1.7
2002	2.9	3.6
2000	NaN	1.5

'Ohio' in frame3.columns
>>>True

2003 in frame3.index
>>>False
```

### 重新索引
```python
obj = pd.Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
obj
>>>
d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64

obj2 = obj.reindex(['a','b','c','d','e'])
obj2
>>>
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64

obj.reindex(['a','b','c','d','e'],fill_value=0)
>>>
a   -5.3
b    7.2
c    3.6
d    4.5
e    0.0
dtype: float64

obj3 = pd.Series(['blue','purple','yellow'],index=[0,2,4])
obj3.reindex(range(6),method='ffill')
>>>
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object

frame = pd.DataFrame(np.arange(9).reshape((3,3)),index=['a','c','d'],columns=['Ohio','Texas','California'])
frame
>>>
Ohio	Texas	California
a	0	1	2
c	3	4	5
d	6	7	8

frame2 = frame.reindex(['a','b','c','d'])
frame2
>>>
Ohio	Texas	California
a	0.0	1.0	2.0
b	NaN	NaN	NaN
c	3.0	4.0	5.0
d	6.0	7.0	8.0

states = ['Texas','Uath','California']
frame.reindex(columns = states)
>>>
Texas	Uath	California
a	1	NaN	2
c	4	NaN	5
d	7	NaN	8

```
