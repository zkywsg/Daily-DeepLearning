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
# 因为错的是class，捕获错误就是捕获了一个class的实例
# 我们可以自己编写函数抛出错误

class FooError(ValueError):
    pass

def foo(s):
    n = int(s)
    if n == 0:
        raise FooError('invaild value: %s'%s)
    return 10/n
foo('0')
>>>
---------------------------------------------------------------------------
FooError                                  Traceback (most recent call last)
<ipython-input-11-4b4551c506aa> in <module>
----> 1 foo('0')

<ipython-input-10-35eddb2011fd> in foo(s)
      2     n = int(s)
      3     if n == 0:
----> 4         raise FooError('invaild value: %s'%s)
      5     return 10/n

FooError: invaild value: 0
```

### 2.调试
- 2.1 print
- 2.2 assert
- 2.3 logging
- 2.4 pdb
```python
# 2.1 print
# %%
def foo(s):
    n = int(s)
    print('n = %d'%n)
    return 10/n
# %%
def main():
    foo('0')
# %%
main()
>>>
# 打印出来错误是因为0
# 明显不是个好方法
n = 0
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-29-263240bbee7e> in <module>
----> 1 main()

<ipython-input-28-e0516f232497> in main()
      1 def main():
----> 2     foo('0')

<ipython-input-26-6a182582d57b> in foo(s)
      2     n = int(s)
      3     print('n = %d'%n)
----> 4     return 10/n

ZeroDivisionError: division by zero
```

```python
# 2.2 assert断言
# 可以用print的地方就可以用assert

# %%
def foo(s):
    n = int(s)
    # n != 0 是True，就继续下去，如果是False就'n is zero!'
    assert n != 0, 'n is zero!'
    return 10 / n
# %%
main()
>>>
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-33-263240bbee7e> in <module>
----> 1 main()

<ipython-input-28-e0516f232497> in main()
      1 def main():
----> 2     foo('0')

<ipython-input-31-daa2ab964953> in foo(s)
      1 def foo(s):
      2     n = int(s)
----> 3     assert n != 0, 'n is zero!'
      4     return 10 / n
# assert语句抛出AssertionError
AssertionError: n is zero!
```

```python
# 2.3 logging
# 不会抛出错误/但是可以输出文件
import logging
# level分为 info/debug/warning/error
logging.basicConfig(level=logging.INFO)
s = '0'
n = int(s)
logging.info('n = %d'%n)
print(10/n)
>>>
INFO:root:n = 0
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-6-20413b32911f> in <module>
----> 1 print(10/n)

ZeroDivisionError: division by zero
```

```python
# 2.4 pdb
# 启动python的调试器pdb，让程序单步运行，可以随时查看运行状态

# %%
# err.py
s = '0'
n = int(s)
print(10 / n)
# %%
```

```python
# 命令行启动
python -m pdb err.py

# 输入n进行单步调试
(pdb) n

# 输入 p (变量名) 查看变量
(pdb) p n
(pdb) p s

# 输入q结束调试
(pdb) q
```

```python
# pdb.set_trace()设置断点

# err.py

s = '0'
n = int(s)
pdb.set_trace() # 运行到这里会自动暂停
print(10 / n)

# p查看变量 c继续执行
(pdb) p n
(pdb) c
```

### 3. 文档测试

```python
# ...省略了一大堆乱七八糟的东西，当出现了xx种情况和文档匹配的时候，就会出现相应的报错
# %%
class Dict(dict):
    '''
    Simple dict but also support access as x.y style.

    >>> d1 = Dict()
    >>> d1['x'] = 100
    >>> d1.x
    100
    >>> d1.y = 200
    >>> d1['y']
    200
    >>> d2 = Dict(a=1, b=2, c='3')
    >>> d2.c
    '3'
    >>> d2['empty']
    Traceback (most recent call last):
        ...
    KeyError: 'empty'
    >>> d2.empty
    Traceback (most recent call last):
        ...
    AttributeError: 'Dict' object has no attribute 'empty'
    '''
    def __init__(self, **kw):
        super(Dict, self).__init__(**kw)

    # def __getattr__(self, key):
    #     try:
    #         return self[key]
    #     except KeyError:
    #         raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value

if __name__=='__main__':
    import doctest
    doctest.testmod()

# %%
```

```python
# 如果把__getattr__()注释掉，会出现这样的测试报错
**********************************************************************
File "__main__", line ?, in __main__.Dict
Failed example:
    d1.x
Exception raised:
    Traceback (most recent call last):
      File "/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/lib/python3.7/doctest.py", line 1329, in __run
        compileflags, 1), test.globs)
      File "<doctest __main__.Dict[2]>", line 1, in <module>
        d1.x
    AttributeError: 'Dict' object has no attribute 'x'
**********************************************************************
File "__main__", line ?, in __main__.Dict
Failed example:
    d2.c
Exception raised:
    Traceback (most recent call last):
      File "/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/lib/python3.7/doctest.py", line 1329, in __run
        compileflags, 1), test.globs)
      File "<doctest __main__.Dict[6]>", line 1, in <module>
        d2.c
    AttributeError: 'Dict' object has no attribute 'c'
**********************************************************************
1 items had failures:
   2 of   9 in __main__.Dict
***Test Failed*** 2 failures.

```

### 4.单元测试
```python
import unittest
# %%
class Dict(dict):

    def __init__(self, **kw):
        super().__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value

# %%
class TestDict(unittest.TestCase):

    def test_init(self):
        d = Dict(a=1, b='test')
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))

    def test_key(self):
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')

    def test_attr(self):
        d = Dict()
        d.key = 'value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'], 'value')

    def test_keyerror(self):
        d = Dict()
        with self.assertRaises(KeyError):
            value = d['empty']

    def test_attrerror(self):
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty
# %%
# 编写测试类，都从unittest.TestCase类
# test开头的就是测试方法，
# unittest.TestCase提供了很多内置的条件判断

# 运行单元测试
if __name__ == '__main__':
    unittest.main()
>>>
# 运行结果
----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK
# %%
```

```python
python -m unittest xxx(文件名)

>>>
# 运行结果
----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK

# 这样可以一次批量运行很多单元测试，并且，有很多工具可以自动来运行这些单元测试。
```
