### 1.错误处理
- 1.1 try except finally
- 1.2 调用栈
- 1.3 记录错误
- 1.4 抛出错误
```python
# 1.1 try机制
try:
    print('try...')
    r = 10/0
    print('result:',r)
except ZeroDivisionError as e:
    print('except:',e)
finally:
    print('finally...')
print('end')
```
\>>>
<br/>
try...
<br/>
except: division by zero
<br/>
finally...
<br/>
end

```python
# 所有的错误类型都继承自BaseException,当使用except的时候不但捕获该类型的错误，子类也一起被捕获
try:
    print('try...')
    r = 10 / int('a')
    print('result:', r)
except ValueError as e:
    print('ValueError')
except UnicodeError as e:
    print('UnicodeError')
# 在这里永远不会捕获到UnicodeError，因为他是ValueError的子类
```
\>>>
<br/>
try...
<br/>
ValueError

```python
# 1.2 调用栈
# 如果错误没有被捕获会一直往上抛，最后被python解释器捕获，打印错误信息，退出程序
# %%
def foo(s):
    return 10/int(s)

def bar(s):
    return foo(s)*2

def main():
    bar('0')
main()
# %%
>>>
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-1-7d9101d864ec> in <module>
      7 def main():
      8     bar('0')
----> 9 main()
# 第一部分：告诉我们第9行有问题

<ipython-input-1-7d9101d864ec> in main()
      6
      7 def main():
----> 8     bar('0')
      9 main()
# 第二部分：告诉我们上一层第8行有问题

<ipython-input-1-7d9101d864ec> in bar(s)
      3
      4 def bar(s):
----> 5     return foo(s)*2
      6
      7 def main():
# 第三部分：告诉我们上一层第5行有问题

<ipython-input-1-7d9101d864ec> in foo(s)
      1 def foo(s):
----> 2     return 10/int(s)
      3
      4 def bar(s):
      5     return foo(s)*2

ZeroDivisionError: division by zero
# 源头：第二行出现了division by zero 的错误
```


```python
# 1.3 记录错误
# 如果不捕获错误，自然可以让python解释器来打印错误堆栈，但程序也结束了！
# 既然我们可以捕获错误，就可以把错误堆栈打印出来，分析错误，同时可以持续运行
# logging模块
import logging
def foo(s):
    return 10/int(s)
def bar(s):
    return foo(s)*2

def main():
    try:
        bar('0')
    except Exception as e:
        logging.exception(e)

main()
print('END')
>>>
ERROR:root:division by zero
Traceback (most recent call last):
  File "<ipython-input-5-a6d7cf5e63df>", line 3, in main
    bar('0')
  File "<ipython-input-4-3ddc15423656>", line 2, in bar
    return foo(s)*2
  File "<ipython-input-3-48e50a3d2c31>", line 2, in foo
    return 10/int(s)
ZeroDivisionError: division by zero
END
# 发现打印完错误之后还会继续执行
# 还可以把错误写进日志里
```

```python
# 1.4 抛出错误

```
