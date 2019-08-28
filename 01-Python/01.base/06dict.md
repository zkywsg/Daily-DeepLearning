
## 字典
- 使用字典
- 访问字典中的值
- 添加键值对


```python
# 字典用{}中放键值对表示
player = {'name':'Kobe','number':24}
# 访问
print(player['name'])
print(player['number'])
print(player)
```

    Kobe
    24
    {'name': 'Kobe', 'number': 24}



```python
# 添加键值对
player['city'] = 'LA'
print(player)
```

    {'name': 'Kobe', 'number': 24, 'city': 'LA'}



```python
# 从空字典添加键值对
player = {}
player
```




    {}




```python
player['name'] = 'Kobe'
player['number'] = 24
player['city'] = 'LA'
player
```




    {'name': 'Kobe', 'number': 24, 'city': 'LA'}




```python
# 修改元素的值
print("Original number:",player['number'])
player['number'] = 8
print("new number:",player['number'])
```

    Original number: 24
    new number: 8



```python
print(player)
# 删除键值对
del player['number']
print(player)
```

    {'name': 'Kobe', 'number': 8}
    {'name': 'Kobe'}



```python
# 遍历字典
user_0 = {
    'username' : 'lauuu',
    'first':'Lau',
    'last': 'Phil',
}
user_0.items()
```




    dict_items([('username', 'lauuu'), ('first', 'Lau'), ('last', 'Phil')])




```python
 for key,value in user_0.items():
    print('\nKey:'+key)
    print('\nValue:'+value)
```

    
    Key:username
    
    Value:lauuu
    
    Key:first
    
    Value:Lau
    
    Key:last
    
    Value:Phil



```python
# 遍历所有的键
for key in user_0.keys():
    print(key)
```

    username
    first
    last



```python
# 按顺序遍历所有的键
for key in sorted(user_0.keys()):
    print(key)
```

    first
    last
    username



```python
# 遍历所有的值
for value in user_0.values():
    print(value)
```

    lauuu
    Lau
    Phil



```python
 # 字典列表
player_01 = {'number':1,'points':10}
player_02 = {'number':2,'points':2}
player_03 = {'number':8,'points':81}

# 组成列表
players = [player_01,player_02,player_03]

for player in players:
    print(player)
```

    {'number': 1, 'points': 10}
    {'number': 2, 'points': 2}
    {'number': 8, 'points': 81}



```python
# 在字典中储存列表
pizza = {
    'crust':'thick',
    'toppings':['mushrooms','extra cheese'],
}

print('You ordered a ' + pizza['crust'] + '-crust pizza' + 'with the following toppings:')
for topping in pizza['toppings']:
    print('\t' + topping)
```

    You ordered a thick-crust pizzawith the following toppings:
    	mushrooms
    	extra cheese



```python
favorite_languages = {
    'jen':['python','ruby'],
    'sarah':['c'],
    'edward':['ruby','go'],
    'phil':['python','haskell'],
}
for name , languages in favorite_languages.items():
    print('\n' + name.title() + "'s favorite language are:'")
    for language in languages:
        print("\t" + language.title())
```

    
    Jen's favorite language are:'
    	Python
    	Ruby
    
    Sarah's favorite language are:'
    	C
    
    Edward's favorite language are:'
    	Ruby
    	Go
    
    Phil's favorite language are:'
    	Python
    	Haskell



```python
# 在字典里储存字典
users = {
    'aeinstein':{
        'first':'albert',
        'last':'einstein',
        'location':'princeton',
    },
    'mcurie':{
        'first':'marie',
        'last':'curie',
        'location':'paris',
    },
}

for username , user_info in users.items():
    print("\nUsername:" + username)
    full_name = user_info['first'] + " " + user_info['last']
    location = user_info['location']
    
    print('\tFull name:' + full_name.title())
    print('\tLocation:' + location.title())
```

    
    Username:aeinstein
    	Full name:Albert Einstein
    	Location:Princeton
    
    Username:mcurie
    	Full name:Marie Curie
    	Location:Paris

