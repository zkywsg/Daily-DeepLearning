### 1.错误处理
- 1.1 try except finally
- 1.2 调用栈
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
