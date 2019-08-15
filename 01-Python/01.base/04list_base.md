
## 列表
- 一系列按照特定顺序排列的元素组成
- [ ]表示列表 ，分割元素
- 访问 索引从0开始


```python
city = ['GuangZhou','ShenZhen','HongKong']
city
```




    ['GuangZhou', 'ShenZhen', 'HongKong']




```python
city[0]
```




    'GuangZhou'




```python
city[0].title()
```




    'Guangzhou'




```python
city[-1]
```




    'HongKong'



## 修改/添加/删除元素
- 修改-直接赋值
- 添加-append末尾添加 insert中间添加
- 删除del pop


```python
city
```




    ['GuangZhou', 'ShenZhen', 'HongKong']




```python
city[0] = 'BeiJing' #修改元素
city
```




    ['BeiJing', 'ShenZhen', 'HongKong']




```python
city.append('BeiJing') #在末尾添加元素
city
```




    ['BeiJing', 'ShenZhen', 'HongKong', 'BeiJing']




```python
city.insert(0,'ShangHai') # 在第一个位置插入
city
```




    ['ShangHai', 'BeiJing', 'ShenZhen', 'HongKong', 'BeiJing']




```python
del city[0] # 删除任何位置的元素
city
```




    ['BeiJing', 'ShenZhen', 'HongKong', 'BeiJing']




```python
city.pop() # 弹出最后一个元素
city
```




    ['BeiJing', 'ShenZhen', 'HongKong']




```python
city.pop(0) # 弹出任何元素
city
```




    ['ShenZhen', 'HongKong']




```python
city.remove('HongKong') # 按值删除
city
```




    ['ShenZhen']



## 组织列表
- sort 永久性排序
- sorted 暂时性排序
- 倒序打印
- 确定列表长度


```python
city = ['GuangZhou','ShenZhen','HongKong']
city.sort()
city
```




    ['GuangZhou', 'HongKong', 'ShenZhen']




```python
city.sort(reverse=True) # 逆序
city
```




    ['ShenZhen', 'HongKong', 'GuangZhou']




```python
city = ['GuangZhou','ShenZhen','HongKong']
sorted(city) # 暂时性排序
```




    ['GuangZhou', 'HongKong', 'ShenZhen']




```python
city # 原列表没变
```




    ['GuangZhou', 'ShenZhen', 'HongKong']




```python
city.reverse() # 调转真的列表（永久）
print(city)
```

    ['HongKong', 'ShenZhen', 'GuangZhou']



```python
len(city) # 列表的长度
```




    3


