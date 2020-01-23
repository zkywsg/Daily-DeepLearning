### 1. 文件读写
- 1.1 读文件
- 1.2 字符编码
- 1.3 写文件
```python
# 1.1 读文件
# open(文件名,标志符)函数
f = open('/xx/xx/xx','r')
# ‘r’表示读
```

```python
# read()一次读取全部内容到内存中，返回一个str
f.read()
```
\>>>'Hello World'

```python
# close()方法关闭文件，文件对象会占用操作系统资源
f.close()
```

```python
# 因为文件读写的时候可能会产生IOError，为了保证close的执行，可以使用try
try:
    f = open('xx/xx','r')
    print(f.read())
finally:
    if f:
        f.close()

# with语句自动调用close
with open('xx/xx','r') as f:
    print(f.read())
```

```python
# 1.2 字符编码
# 读取非utf-8的文本文件/需要给open传递encoding参数
# 比如gbk编码的中文
f = open('xx/xx.txt','r',encoding='gbk')
f.read()
```
\>>>'测试'

```python
# 当夹杂这一些非法字符的时候，可能会遇到UnicodeDecodeError,这时候就要用到error参数
f = open('xx/xx.txt','r',encoding='gbk',errors='ignore')
```

```python
# 1.3 写文件
# 和读文件的区别就是把标志符换成w
f = open('xx/xx.txt','w')
```

### 2. StringIO和BytesIO
```python
# StringIO
# 就是在内存中对String进行读写
from io import StringIO
f = StringIO()
f.write('hello')
>>>5
f.write(' ')
>>>1
f.write('world')
>>>5
print(f.getvalue())
```
\>>>hello world

```python
# 读取
from io import StringIO
f = StringIO('Hello\nWorld')
while True:
    s = f.readline()
    if s == '':
        break
    print(s.strip())
```
>>>
<br/>
Hello
<br/>
World

```python
# BytesIO 实现在内存中二进制流的读写
from io import BytesIO
f = BytesIO()
# 写入的不是str，而是经过utf-8编码的字节流
f.write('中文'.encode('utf-8'))
>>>6
print(f.getvalue())
```
\>>>b'\xe4\xb8\xad\xe6\x96\x87'

```python
from io import BytesIO
f = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
f.read()
```
\>>>b'\xe4\xb8\xad\xe6\x96\x87'

### 3. 操作文件和目录
```python
import os
# 操作系统类型 'posix'代表linux unix macos 'nt'就是windows
os.name
```
\>>>'posix'

```python
# 具体的系统信息
os.uname()
```
\>>>
posix.uname_result(sysname='Darwin', nodename='LaudeMacBook-Pro.local', release='19.0.0', version='Darwin Kernel Version 19.0.0: Thu Oct 17 16:17:15 PDT 2019; root:xnu-6153.41.3~29/RELEASE_X86_64', machine='x86_64')

```python
# 操作系统定义的环境变量
os.environ
```

```python
# 查看当前绝对路径
os.path.abspath('.')
```
\>>>/xxx/xxx/Day11.md

```python
# 在某个目录下创建新的目录,首先把新目录的路径完整表示出来
os.path.join('xx/xx','new')
```
\>>>'xx/xx/new'

```python
# 创建新目录
os.mkdir('xx/xx/new')

# 删除目录
os.rmdir('xx/xx/new')
```

```python
# 拆分路径 os.path.split()
# 后一部分一定是最后级别的目录或文件
os.path.split('xx/xx/xx.txt')
```
\>>>('xx/xx', 'xx.txt')

```python
# os.path.splitext(),让你得到文件扩展名
os.path.splitext('/path/file.txt')
```
\>>>('/path/file', '.txt')

```python
# 对文件重命名
os.rename('text.txt','text.py')

# 删掉文件
os.remove('test.py')
```

### 4.序列化
- 4.1 dump和dumps
- 4.2 loads
- 4.3 json
```python
# 4.1 dump和dumps
# 在程序运行的时候，所有的变量都是在内存里
d = dict(name='Bob',age=20,score=88)

# 虽然可以随时修改变量，但是一旦结束，变量占用的内存就被操作系统全部回收
# 没有存入磁盘的话，下次运行程序就会发现变量仍然没有修改
# 我们把变量从内存中变成可储存的过程叫做序列化
# 序列化后写入磁盘或通过网络存在别的机器上
```

```python
# 把一个对象写入文件
import pickle
d = dict(name='Bob',age=20,score=88)
# dumps()把对象序列变成bytes，然后就可以把它写入文件
pickle.dumps(d)
```
\>>>b'\x80\x03}q\x00(X\x04\x00\x00\x00nameq\x01X\x03\x00\x00\x00Bobq\x02X\x03\x00\x00\x00ageq\x03K\x14X\x05\x00\x00\x00scoreq\x04KXu.'

```python
# dump()把对象序列化后直接写入file-like-object
f = open('dump.txt','wb')
pickle.dump(d,f)
f.close()
```
\>>>查看dump.txt里面乱七八糟的内容，就是保存的对象内部信息

```python
# 4.2loads
# 把对象从磁盘读到内存/可以用loads()反序列化/也可以用load()方法从file-like-object反序列化出对象
f = open('dump.txt','rb')
d = pickle.load(f)
f.close()
d
```
\>>>{'age': 20, 'score': 88, 'name': 'Bob'}

```python
# pickle只能在python内使用
# JSON表示的对象就是标准的JavaScript语言的对象
# 把python对象变成一个json
import json
d = dict(name='Bob', age=20, score=88)
json.dumps(d)
```
\>>>'{"name": "Bob", "age": 20, "score": 88}'

```python
# loads()
json_str = '{"age": 20, "score": 88, "name": "Bob"}'
# 变成python对象
json.loads(json_str)
```
\>>>{'age': 20, 'score': 88, 'name': 'Bob'}
