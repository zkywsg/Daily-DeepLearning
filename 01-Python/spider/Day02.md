### urllib.error
- URLError类来自于urllib库的error模块
```python
from urllib import request,error
try:
    response = request.urlopen('http://cuiqingcai.com/index.htm')
except error.URLError as e:
    print(e.reason)
```
\>>>Not Found

- HTTPError:URLError的子类/有3个属性
  - code:返回HTTP状态码/比如404
  - reason：返回错误原因
  - headers：返回请求头

```python
from urllib import request,error

try:
    response = request.urlopen('http://cuiqingcai.com/index.htm')
except error.HTTPError as e:
    print(e.reason,e.code,e.headers,sep='\n')
```

```
Not Found
404
Server: nginx/1.10.3 (Ubuntu)
Date: Wed, 05 Feb 2020 15:21:54 GMT
Content-Type: text/html; charset=UTF-8
Transfer-Encoding: chunked
Connection: close
Set-Cookie: PHPSESSID=gk6tgskpefsdslbcpkcp5vhm36; path=/
Pragma: no-cache
Vary: Cookie
Expires: Wed, 11 Jan 1984 05:00:00 GMT
Cache-Control: no-cache, must-revalidate, max-age=0
Link: <https://cuiqingcai.com/wp-json/>; rel="https://api.w.org/"
```


- 可以先捕获子类错误/再去捕获父类的错误
- 这样的逻辑/是较好的异常处理写法
```python
from urllib import request,error

try:
    response = request.urlopen('http://cuiqingcai.com/index.htm')
except error.HTTPError as e:
    print(e.reason,e.code,e.headers,sep='\n')
except error.URLError as e:
    print(e.reason)
else:
    print('Request Successfully')
```

- 有时候reason返回的是对象
- 抛出timeout异常

```python
import socket
import urllib.request
import urllib.error

try:
    response = urllib.request.urlopen('https://www.zhihu.com',timeout=0.01)
except urllib.error.URLError as e:
    print(type(e.reason))
    if isinstance(e.reason,socket.timeout):
        print('TIME OUT')
```
\>>>
<br/>
<class 'socket.timeout'>
<br/>
TIME OUT
