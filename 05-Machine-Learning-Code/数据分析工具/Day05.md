### 丢弃指定轴上的项

```python
import pandas as pd
import numpy as np
obj = pd.Series(np.arange(5.),index=['a','b','c','d','e'])
new_obj = obj.drop('c')
new_obj
>>>
a    0.0
b    1.0
d    3.0
e    4.0
dtype: float64

obj.drop(['d','c'])
>>>
a    0.0
b    1.0
e    4.0
dtype: float64
```
- 对于DataFrame/可以删除任意轴上的索引值
```python
data = pd.DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],
            columns=['one','two','three','four'])
data.drop(['Colorado','Ohio'])
>>>
one	two	three	four
Utah	8	9	10	11
New York	12	13	14	15

data.drop('two',axis=1)
>>>
one	three	four
Ohio	0	2	3
Colorado	4	6	7
Utah	8	10	11
New York	12	14	15

data.drop('Ohio',axis=0)
>>>
one	two	three	four
Colorado	4	5	6	7
Utah	8	9	10	11
New York	12	13	14	15
```

### 索引/选取和过滤
```python
obj = pd.Series(np.arange(4.),index=['a','b','c','d'])
obj['b']
>>>1.0

obj[1]
>>>1.0

obj[2:4]
>>>
c    2.0
d    3.0
dtype: float64

obj[['b','a','d']]
>>>
b    1.0
a    0.0
d    3.0
dtype: float64

obj[[1,3]]
>>>
b    1.0
d    3.0
dtype: float64

obj[obj<2]
>>>
a    0.0
b    1.0
dtype: float64

obj['b':'c']
>>>
b    1.0
c    2.0
dtype: float64

obj['b':'c'] = 5
obj
>>>
a    0.0
b    5.0
c    5.0
d    3.0
dtype: float64
```
```python
data = pd.DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],
                    columns=['one','two','three','four'])
data
>>>
one	two	three	four
Ohio	0	1	2	3
Colorado	4	5	6	7
Utah	8	9	10	11
New York	12	13	14	15

data['two']

data[['three','one']]
>>>
three	one
Ohio	2	0
Colorado	6	4
Utah	10	8
New York	14	12

data[:2]
>>>
one	two	three	four
Ohio	0	1	2	3
Colorado	4	5	6	7

data[data['three']>5]
>>>
one	two	three	four
Colorado	4	5	6	7
Utah	8	9	10	11
New York	12	13	14	15

data < 5
>>>
one	two	three	four
Ohio	True	True	True	True
Colorado	True	False	False	False
Utah	False	False	False	False
New York	False	False	False	False

data[data < 5] = 0
data
>>>
one	two	three	four
Ohio	0	0	0	0
Colorado	0	5	6	7
Utah	8	9	10	11
New York	12	13	14	15
```

- ix被代替了
- loc根据行标签
- iloc根据行号
```python
data.loc['Colorado',['two','three']]
>>>
two      5
three    6
Name: Colorado, dtype: int64

data.iloc[2]
>>>
one       8
two       9
three    10
four     11
Name: Utah, dtype: int64

data.loc[:'Utah','two']
>>>
Ohio        0
Colorado    5
Utah        9
Name: two, dtype: int64

data.loc[data.three>5][:3]
>>>
one	two	three	four
Colorado	0	5	6	7
Utah	8	9	10	11
New York	12	13	14	15
```

### 算术运算和数据对齐

```python

```
