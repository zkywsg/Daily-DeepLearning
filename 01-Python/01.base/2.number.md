
- 创建数值对象
- 更新数值对象
- 删除数值对象


```python
Int = 1
print(Int)
```

    1



```python
Int += 1
print(Int)
```

    2



```python
del Int
print(Int)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-33a5e8dd0e19> in <module>
    ----> 1 del Int
          2 print(Int)


    NameError: name 'Int' is not defined


- ~取反
- & 按位与
- | 或
- ^ 异或
- << 左移
- \>\> 右移


```python
30 & 45
```




    12




```python
30 | 45
```




    63




```python
~12
```




    -13




```python
30 << 1
```




    60




```python
30 >> 1
```




    15



## 标准类型函数
- str() 数字转化为字符串
- type() 返回数字对象的类型


```python
str(0xFF)
```




    '255'




```python
type(0xFF)
```




    int



## 运算符模块operator


```python
import operator
operator.gt(1,2) #1>2?
```




    False




```python
operator.ge(1,2) #1>=2?
```




    False




```python
operator.eq(1,1) #1==1?
```




    True




```python
operator.le(1,2) # 1<=2?
```




    True




```python
operator.lt(1,2) # 1<2?
```




    True



## 数字类型转化


```python
int(4.23)
```




    4




```python
float(3)
```




    3.0




```python
complex(2.4,-8) # 复数
```




    (2.4-8j)


