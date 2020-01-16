### 1. if

```python
# 通过缩进进行分段 掌握if else 和 elif
age = 3
if age >= 18:
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('kid')
```
\>>>kid


###  2. 循环
- for循环的基本用法
- while循环的基本用法
- break
- continue

```python
# for...in ... 把列表或者元组中的元素迭代出来
names = ['sb','zz','ni']
for name in names:
    print(name)
```
\>>>
<br/>
sb
<br/>
zz
<br/>
ni

```python
# 使用range函数，range(5)表示0-4
for x in range(5):
    print(x)
```
\>>>0
1
2
3
4

```python
# while循环的基本用法
# 计算1-10的和
sum = 0
n = 1
while n<=10:
    sum = sum + n
    n = n + 1
print(sum)
```
\>>>55

```python
# break直接跳出循环
sum = 0
n = 1
while n<=100:
    if n>10:
        break
    sum = sum + n
    n = n + 1
print(sum)
```
\>>>55

```python
# continue 跳过循环中的本次
n = 0
while n<5:
    n = n + 1
    if n==3:
        continue
    print(n)
```
\>>>1
2
4
5
