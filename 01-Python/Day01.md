### 1. Hello World

```python
print("Hello World!")
```
\>>> Hello World!


### 2. 变量
- 2.1 命名
 - 变量名：包含字母/数字/下划线，只能字母和下划线开头，数字不能打头。e.g. tensor_1是正确的，1_tensor是错误的
 - 不能包含空格 e.g. tensor 1是错误的
 - 避开python的关键字和函数名用作变量名，也不要用python中有特殊用途的单词，e.g. print
 - 尽量简洁又有描述性，e.g. student_name就比s_n要更好
- 2.2 变量赋值
 - 不需要类型声明
 - 每个变量都必须赋值，赋值以后变量才会被创建

```python
# 2.1 命名
# 2.2 变量赋值
number = 10 # 整型变量
distance = 10.0 # 浮点型变量
name = 'sb' # 字符串
print(number)
print(distance)
print(name)
```
\>>>10

10.0

sb


- 2.3 多个变量赋值
 - 同时为多个变量赋值
 - 为多个对象指定多个变量

```python
# 2.3.1 同时为多个变量赋值
a = b = c = 1
print(a,b,c)
```
\>>>1 1 1


```python
# 2.3.2 为多个对象指定多个变量
a, b, c = 1, 1.0, 'sb'
print(a,b,c)
```
\>>>1

1.0

sb


### 3. 字符串
 - 3.1 修改字符串的大小写
 - 3.2 拼接字符串
 - 3.3 制表符和换行符
 - 3.4 删除空白
 - 3.5 编码问题
 - 3.6 格式化输出
 - 3.7 索引

```python
# 3.1 修改字符串的大小写
# 3.1.1 让首字母大写 title()函数
name = 'clear love'
print(name.title())
```
\>>>Clear Love


```python
# 3.1.2 让所有字母变成大写 upper()函数
print(name.upper())
```
\>>>CLEAR LOVE


```python
# 3.1.3 让所有字母变成小写 lower()函数
print(name.lower())
```
\>>>clear love


```python
# 3.2 通过+来合并字符串
first_name = 'clear'
last_name = 'love'
full_name = first_name + " " + last_name
print('Hello ' + full_name.title() + '!')
```
\>>>Hello Clear Love!


```python
# 3.3.1 制表符\t
print("sb")
print("\tsb")
```
\>>>sb

\>>>&emsp;sb

```python
# 3.3.2 换行符\t
print("NT\nSB\nNC")
```
\>>>NT

SB

NC


```python
# 3.4.1 删除左边的空白 lstrip()函数
your_name = ' sb '
print(your_name.lstrip())

# 3.4.2 删除右边的空白 rstrip()函数
print(your_name.rstrip())

# 3.4.3 删除两边的空白 strip()函数
print(your_name.strip())
```
\>>> sb

\>>>&emsp;sb

\>>>sb


```python
# 3.5 编码问题
# 我们要先了解几种编码，ASCII:一个字节，包括了一些数字和英文字母
# Unicode:两个字节，由于一个字节被一些数字和英文字母用完了，那么汉字在一个字节的情况下就会出现乱码，所以两个字节的Unicode就诞生了
# UTF-8变长编码：用一个字节表示英文字母，三个字节表示汉字，4-6个字节表示生僻字
# Python3中字符串使用Unicode编码

# ord()函数：获得字符的整数表示
# chr()函数：获得整数的字符表示
print(ord('A'))
print(chr(66))
```
\>>>65

B


```python
# 编码
print('A'.encode('ascii'))
print('哈哈'.encode('utf-8'))
```
\>>>b'A'

b'\xe5\x93\x88\xe5\x93\x88'

```python
# 解码
print(b'ABC'.decode('ascii'))
print(b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8'))
```
\>>>ABC

中文

```python
# 若有错误
b'\xe4\xb8\xad\xff'.decode('utf-8')
```
\>>>---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
<ipython-input-40-cd8de1b11dcd> in <module>
----> 1 b'\xe4\xb8\xad\xff'.decode('utf-8')

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 3: invalid start byte

```python
# 传入参数errors='ignore' 忽略错误
b'\xe4\xb8\xad\xff'.decode('utf-8',errors='ignore')
```
\>>>'中'


```python
# len()函数计算字符个数或字节数
print(len('中文')) #字符个数
print(len(b'\xe4\xb8\xad\xe6\x96\x87')) #字节数
```
\>>>2

6

```python
# 3.6 格式化输出
# 3.6.1 %d 整数 %f 浮点数 %s 字符串 %x 十六进制数
'Hello, %s' %'world'
```
\>>>'Hello, world'


```python
# 3.6.2 遇到需要表示普通字符%的时候
print('%d %%'%10)
```
\>>>10 %

```python
# 3.6.3 format()函数对应{0},{1}······
# {1:.1f} -> .1f精确到小数点后一位
'hello {0},you get {1:.1f} point'.format('sb',0.000)
```
\>>>'hello sb,you get 0.0 point'


# 3.7 索引
# 从左到右从0开始
# 从右到左从-1开始

str = 'Hello world'

# 完整
print(str)

# 第一个字符
print(str[0])

# 第2-5个
print(str[1:5])

# 第3个以后
print(str[2:])

# 最后一个
print(str[-1])

# 输出两次
print(str*2)
# %% markdown
# ### 4.数字和运算符
# - 4.1 加减乘除/乘方/取模/取整数
# - 4.2 比较运算符
# - 4.3 位运算符
# - 4.4 逻辑运算符
# %%
# 4.1 整数的加减乘除/乘方/取模/取整除

# 加法
print(2+3)

#减法
print(3-2)

# 乘法
print(2*3)

# 除法
print(3/2)

# 乘方
print(3 ** 2)

# 取模
print(5%2)

# 取整除/向下取整
print(3/2)
# %%
# 4.2 比较运算符

# ==相等返回true
print(1 == 2)

# != 不相等返回true
print(1 != 2)

# <:小于 >:大于 <=:小于等于 >=:大于等于
print(1 > 2)
print(1 < 2)
print(1 >= 1)
print(1 <= 2)
# %%
# 4.3 位运算符

# 与运算 都是1的时候才为1
print('1&0:',1&0)
print('1&1:',1&1)
print('0&0:',0&0)
print('---------------------------')

# 或运算 有1就得1
print('1|0:',1|0)
print('1|1:',1|1)
print('0|0:',0|0)
print('---------------------------')

# 异或运算 不相同才得1
print('1^1:',1^1)
print('1^0:',1^0)
print('0^0:',0^0)
print('---------------------------')

# 左移运算符 高位丢弃/低位补0 2:0000 0010 左移一位 4:0000 0100
print('2<<1:',2 << 1)

# 右移运算符 地位丢弃/高位补0 2:0000 0010 右移一位 1:0000 0001
print('2>>1:',2 >> 1)
# %%
# 4.4 逻辑运算符 布尔‘与‘’或‘’非‘

# and ： x and y,若x为false，返回false，否则返回y的计算值
a = False
b = 2
print('a=False,b=2,a and b:',a and b)
print('-----------------------------')

a = 1
print('a=1,b=2,a and b:',a and b)
print('-----------------------------')

# or : x or y ,若x不是0，则返回x的值，否则返回y的计算值
print('a=1,b=2,a or b:',a or b)
print('-----------------------------')

a = 0
print('a=0,b=2,a or b:',a or b)
print('-----------------------------')

# not : not x
print('a=0,not a:',not a)
