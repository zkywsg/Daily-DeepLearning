### 创建ndarray

```python
import numpy as np
data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)
print(arr1)
>>>
[6.  7.5 8.  0.  1. ]
```



### 通过数组转换生成ndarray

```python
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
print(arr2)
>>>
[[1 2 3 4]
 [5 6 7 8]]

print(arr2.ndim)
>>>
2

print(arr2.shape)
>>>
>>>(2, 4)

print(arr1.dtype)
>>>dtype('float64')

print(arr2.dtype)
>>>dtype('int64')
```



### zeros/ones/empty

```python
np.zeros(10)
>>>array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

np.zeros((3,6))
>>>array([[0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])

np.empty((2,3,3)) # 没有任何具体值的数组
>>>
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])
```



### arange方法

```python
np.arange(15)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```

