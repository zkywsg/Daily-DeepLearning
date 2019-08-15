
## 字符串 
- 用单引号或者双引号


```python
"This is a string"
```




    'This is a string'




```python
'This is a string'
```




    'This is a string'



## 修改大小写
- title() 首字母大写
- upper() 大写
- lower() 小写


```python
name = 'aka sb'
name.title()
```




    'Aka Sb'




```python
name.upper()
```




    'AKA SB'




```python
name.lower()
```




    'aka sb'



## 一些字符串操作
- 用+号合并
- 制表符或者换行符添加空白 \t \n


```python
first_name = 'aka'
last_name = 'sb'
```


```python
full_name = first_name + " " + last_name
full_name
```




    'aka sb'




```python
"GuangDong"
```




    'GuangDong'




```python
print("\tGuangDong")
```

    	GuangDong



```python
print("GuangZhou\nHongKong\nShenZhen")
```

    GuangZhou
    HongKong
    ShenZhen


## 删除空白
- rstrip()
- lstrip()
- strip()


```python
Your_name = ' sb '
Your_name
```




    ' sb '




```python
Your_name.rstrip()
```




    ' sb'




```python
Your_name.lstrip()
```




    'sb '




```python
Your_name.strip()
```




    'sb'


