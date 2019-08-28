
## 遍历整个列表


```python
citys = ['Guangzhou','Shenzhen','HongKong']
citys
```




    ['Guangzhou', 'Shenzhen', 'HongKong']




```python
# 遍历打印整个列表
for city in citys:
    print(city)
```

    Guangzhou
    Shenzhen
    HongKong


## range()函数


```python
# 打印1<= vlaue < 5的数字
for value in range(1,5):
    print(value)
```

    1
    2
    3
    4



```python
# 打印1<= value < 6的数字
for value in range(1,6):
    print(value)
```

    1
    2
    3
    4
    5



```python
# 把1<= value < 6的数字转化成一个列表集合
numbers = list(range(1,6))
print(numbers)
```

    [1, 2, 3, 4, 5]


- 使用list将range的结果转化成列表 
- range函数可以指定步长


```python
numbers = list(range(1,6))
numbers
```




    [1, 2, 3, 4, 5]




```python
# 步长为2，即每隔两个挑一个数字
even_numbers = list(range(2,11,2))
even_numbers
```




    [2, 4, 6, 8, 10]



- 打印1-10的平方 


```python
squares = []
for value in range(1,11):
    squares.append(value**2)
squares
```




    [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]



## 对数字列表的简单统计
- min函数
- max函数
- sum函数


```python
digits = [1,2,3,4,5,6,7,8,9,0]
min(digits)
```




    0




```python
max(digits)
```




    9




```python
sum(digits)
```




    45



## 切片
- 访问部分list
- 遍历切片内容


```python
players = ['kb','kd','sc','lbj','kt']
# 下标从0开始，选择下表为0，1，2的元素
players[0:3]
```




    ['kb', 'kd', 'sc']




```python
# 下标为1，2，3
players[1:4]
```




    ['kd', 'sc', 'lbj']




```python
# 下标为0，1，2，3
players[:4]
```




    ['kb', 'kd', 'sc', 'lbj']




```python
# 下标为：2，3，4
players[2:]
```




    ['sc', 'lbj', 'kt']




```python
# 下标为 2，3，4
players[-3:]
```




    ['sc', 'lbj', 'kt']




```python
# 遍历切片内容
for player in players[:3]:
    print(player.title())
```

    Kb
    Kd
    Sc


## 元组
- 定义元组
- 遍历元组的所有值
- 修改元素变量
- Python将不能修改的值称为不可变的，而不可变的列表称为元组


```python
# 定义元组
# 使用圆括号创立，访问元素方法和列表一样
dims = (100,20)
print("dims0:",dims[0])
print("dims1:",dims[1])
```

    dims0: 100
    dims1: 20



```python
# 如果尝试修改元组的元素，会出现报错 'tuple' object does not support item assignment
dims[0] = 0
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-22-95c946d5ffee> in <module>
          1 # 如果尝试修改元组的元素，会出现报错
    ----> 2 dims[0] = 0
    

    TypeError: 'tuple' object does not support item assignment



```python
# 遍历方式和列表相同
for dim in dims:
    print(dim)
```

    100
    20



```python
# 修改元组方法
# ！不能直接修改某个元素
# 但是可以直接修改整个储存元组的变量
print("Original dims:")
for dim in dims:
    print(dim)
```

    Original dims:
    100
    20



```python
dims = (400,100)
print("New dims:")
for dim in dims:
    print(dim)
```

    New dims:
    400
    100



```python

```
