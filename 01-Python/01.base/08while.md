
## 函数input()
- 让程序暂停，等待用户输入一些文本。


```python
massage = input("Tell me something,and I will repeat it back to you:")
print(massage)
```

    Tell me something,and I will repeat it back to you:hello world
    hello world


- 使用int()获取输入的数值


```python
age = input("how old are you?")
```

    how old are you?22



```python
# age->string
age = int(age)
age >= 18
```




    True



## 使用while循环


```python
current_number = 1
while current_number <= 5:
    print(current_number)
    current_number += 1
```

    1
    2
    3
    4
    5


- 使用break推出循环


```python
prompt = "\nPlease enter the name of a city you have visited:"
prompt += "\n(Enter 'quit' when you are finished.) "
while True:
    city = input(prompt)
    #  一直到输入quit才终止
    if city == 'quit':
        break
    else:
        print("I‘d love to go to " + city.title())
```

    
    Please enter the name of a city you have visited:
    (Enter 'quit' when you are finished.) New York
    I‘d love to go to New York
    
    Please enter the name of a city you have visited:
    (Enter 'quit' when you are finished.) San Francisco
    I‘d love to go to San Francisco
    
    Please enter the name of a city you have visited:
    (Enter 'quit' when you are finished.) quit


- 在循环中使用continue


```python
current_number = 0
while current_number < 10:
    current_number += 1
    # 只要是偶数就跳过当前的循环 进行下一次的循环
    if current_number % 2 == 0:
        continue
    
    print(current_number)
```

    1
    3
    5
    7
    9


## 使用while循环来处理列表和字典
- 在列表之间移动元素
- 删除包含特定值的所有列表元素
- 使用用户输入来填充字典


```python
#在列表之间移动元素
unconfirmed_users = ['alice','brian','candace']
confirmed_users = []

while unconfirmed_users:
    current_user = unconfirmed_users.pop()
    print("verifying user:" + current_user.title())
    confirmed_users.append(current_user)

# 显示所有已经验证的用户
print("\nThe following users have been confirmed:")
for confirmed_user in confirmed_users:
    print(confirmed_user.title())
```

    verifying user:Candace
    verifying user:Brian
    verifying user:Alice
    
    The following users have been confirmed:
    Candace
    Brian
    Alice



```python
# 删除包含特定值的所有列表元素
pets = ['dog','cat','dog','goldfish','cat','rabbit','cat']
print(pets)

while 'cat' in pets:
    pets.remove('cat')
    
print(pets)
```

    ['dog', 'cat', 'dog', 'goldfish', 'cat', 'rabbit', 'cat']
    ['dog', 'dog', 'goldfish', 'rabbit']



```python
# 使用用户输入来填充字典
responses = {}

polling_active = True
while polling_active:
    name = input("\nwhat is your name?")
    response = input("which mountain would you like to climb someday?")
    
    responses[name] = response
    
    repeat = input("would you like to let another person respond(yes/no))")
    if repeat == 'no':
        polling_active = False
        
print("\n---Poll Results---")
for name,response in responses.items():
    print(name + "would like to climb " + response + ".")
```

    
    what is your name?a
    which mountain would you like to climb someday?yes
    would you like to let another person respond(yes/no))yes
    
    what is your name?b
    which mountain would you like to climb someday?no
    would you like to let another person respond(yes/no))no
    
    ---Poll Results---
    awould like to climb yes.
    bwould like to climb no.



```python

```
