### 1. 字典
- 1.1 字典的定义
- 1.2 访问某个关键值对应的值
- 1.3 添加新的键值对
- 1.4 修改字典
- 1.5 删除字典元素
- 1.6 遍历键值对
- 1.7 遍历字典中所有的键
- 1.8 遍历字典中所有的值
- 1.9 列表中有字典
- 1.10 字典中有列表
- 1.11 字典中有字典

```python
# 1.1 字典的定义：是一种可变容器，可以储存任意类型对象 形如：d = {key:value}
d = {'a':1,'b':2,'c':3}
d
```
\>>>{'a': 1, 'b': 2, 'c': 3}

```python
# 1.2 访问某个关键值对应的值 就像访问数组一样，[]内的下标是字典的键值
print(d['a'])
```
\>>>1

```python
# 访问外星人alien_0的颜色和点数/如果玩家射杀这个外星人，就可以获得相应的点数
alien_0 = {'color':'green','points':5}
new_points = alien_0['points']
print('You just earned ' + str(new_points) + ' points!')
```
\>>>You just earned 5 points!

```python
# 1.3 添加新的键值对
print('添加前的字典:',d)
d['d'] = 4
d['e'] = 5
print('添加后的字典:',d)
```
\>>>添加前的字典: {'a': 1, 'b': 2, 'c': 3}

&emsp;添加后的字典: {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

```python
# 1.4 修改字典
print('修改前的字典:',d)
d['a'] = 666
print('修改后的字典:',d)
```
\>>>修改前的字典: {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

&emsp;修改后的字典: {'a': 666, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

```python
# 1.5 删除字典元素 del
print('删除前的字典:',d)
# 删除元素c
del d['c']
print('删除c后的字典',d)
```
\>>>删除前的字典: {'a': 666, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

&emsp;删除c后的字典 {'a': 666, 'b': 2, 'd': 4, 'e': 5}

```python
# 清空所有条目
d.clear()
print('清空所有条目后的字典:',d)
```
\>>>清空所有条目后的字典: {}

```python
# 1.6 遍历键值对
user = {'username':'sbzz','first':'sb','last':'zz'}
# 用 for key,value in xx.items():
for k,v in user.items():
    print('\nkey:',k)
    print('\nvalue',v)
```
\>>>
key: username

value sbzz

key: first

value sb

key: last

value zz


```python
# 1.7 遍历字典中所有的键
# for key in xx.keys():
for k in user.keys():
    print(k)
```
\>>>username

first

last

```python
# 按顺序遍历字典中所有的键 sorted()
for k in sorted(user.keys()):
    print(k)
```
\>>>first

last

username

```python
# 1.8 遍历字典中所有的值
# for value in xx.values():
for v in user.values():
    print(v)
```
\>>> sbzz

sb

zz

```python
# 1.9 列表中有字典
# 每个外星人用字典表示它的各种属性，用一个大的列表扩起来，组成一个外星人列表
alien_0 = {'color':'green','point':5}
alien_1 = {'color':'red','point':3}
alien_2 = {'color':'yellow','point':2}

aliens = [alien_0,alien_1,alien_2]

# 遍历
for alien in aliens:
    print(alien)
```
\>>>{'color': 'green', 'point': 5}

{'color': 'red', 'point': 3}

{'color': 'yellow', 'point': 2}

```python
# 1.10 字典中有列表
# 人和自己喜欢的编程语言是一个键值对，喜欢的语言不止一种，可以有多种语言构成一个列表
favorite_languages = {
    'sb':['python','c++'],
    'zz':['java','php']
}
for name,languages in favorite_languages.items():
    print('\n' + name.title() + '\'s favorite languages are:')
    for language in languages:
          print('\t'+language.title())
```
\>>>

Sb's favorite languages are:

    Python

    C++

Zz's favorite languages are:

    Java

    Php

```python
# 1.11 字典中有字典
# 有很多的用户 每个用户有first name和last name ，所以用户是一个大的字典，每个小字典是{名字：全名}
user = {
    'aeinstein':{
        'first':'albert',
        'last':'einstein'
    },
    'mcurie':{
        'first':'marie',
        'last':'curie'
    },
}

for username,fullname in user.items():
    print('\nUsername:'+username)
    full_name = fullname['first'] + ' ' + fullname['last']
    print('Full name:' + full_name.title())
```
\>>>
Username:aeinstein

Full name:Albert Einstein

Username:mcurie

Full name:Marie Curie

###  2. Set
- 2.1 建立一个set
- 2.2 重复元素自动过滤
- 2.3 添加元素 add()
- 2.4 删除元素 remove()

```python
# 2.1 建立一个set
# set和dict类似，但只有key没有value，并且key不能重复
s = set([1,2,3])
s
```
\>>>{1, 2, 3}

```python
# 2.2 重复元素自动过滤
# 传入的是列表
s = set([1,1,1,2,2,3,3,3,3])
s
```
\>>>{1, 2, 3}

```python
# 2.3 添加元素 add()函数
s.add(4)
s
```
\>>>{1, 2, 3, 4}

```python
# 2.4 删除元素 remove()函数
s.remove(4)
s
```
\>>>{1, 2, 3}
