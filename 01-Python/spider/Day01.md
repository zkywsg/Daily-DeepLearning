### 1.一些概念
- 获取网页
  - 获得网页的源代码/源代码包括一些网页的有用信息/把源代码获取下来/就可以提取想要的信息
  - 向网站的服务器发送一个请求/返回的相应体便是网页源代码

- 提取信息
  - 网页源代码获得之后/分析网页源代码/最普遍的方法就是用正则表达式提取/但是也容易出错
  - 网页有一定规则/网页节点属性/css选择器/XPath来提取网页信息/如Beautiful Soup/pyquery/lxml

- 保存
  - 可以保存为txt/json文本/也可以保存到数据库如MySQL
- 自动化


### 2.urllib.request

- read()得到返回的网页内容
```python
import urllib.request

# 抓取整个网页的源代码
res = urllib.request.urlopen('https://www.zhihu.com')
# read()可以得到返回的网页内容
print(res.read().decode('utf-8'))
```

```python
# 查看响应类型
import urllib.request

res = urllib.request.urlopen('https://www.zhihu.com')
print(type(res))
```
\>>><class 'http.client.HTTPResponse'>

<br/>

- status():可以得到返回结果的状态码/200表示请求成功
- getheaders():得到响应的头信息
- getheader():通过传递参数获得对应的值
```python
import urllib.request

res = urllib.request.urlopen('https://www.zhihu.com')
print(res.status)
print(res.getheaders())
print(res.getheader('Server'))
```
\>>>
<br/>
200
<br/>
[('Server', 'Tengine'), ('Content-Type', 'text/html; charset=utf-8'), ···
<br/>
Tengine
<br/>
<br/>
- Request():urlopen()方法可以实现最基本请求的发起/需要加入更多信息用Request()
- urlopen()仍然用在发送请求/参数变成了一个Request类型的对象
```python
import urllib.request

request = urllib.request.Request('https://www.zhihu.com')
response = urllib.request.urlopen(request)
print(response.read().decode('utf-8'))
```

- Request()构造方法
```python
class urllib.request.Request(url,data=None,headers={},origin_req_host=None,unverifiable=False,method=None)
```

- url:用于请求URL/必传参数
- data:必须是bytes/如果传入字典/用urllib.parse的urlencode()编码
- headers:字典/请求头/最常用的方法就是通过修改User-Agent来伪装浏览器
- origin_req_host:是请求方的host名称或者IP地址
- unverifiable：表示这个请求是否无法验证/就是用户有没足够的权限来接收这个请求的结果
- method:是一个字符串/说明请求使用的方法GET/POST等

```python
from urllib import request,parse
# %%
url = 'http://httpbin.org/post'
headers = {
  'User-Agent':'Mozilla/5.0 (Macintosh;Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
  'Host':'httpbin.org'
}
dict = {
  'name':'sb'
}
data = bytes(parse.urlencode(dict),encoding='utf8')
req = request.Request(url=url,data=data,headers=headers,method='POST')
response = request.urlopen(req)
print(response.read().decode('utf-8'))
# %%
```
- 成功设置了data/headers/method
```
{
  "args": {},
  "data": "",
  "files": {},
  "form": {
    "name": "sb"
  },
  "headers": {
    "Accept-Encoding": "identity",
    "Content-Length": "7",
    "Content-Type": "application/x-www-form-urlencoded",
    "Host": "httpbin.org",
    "User-Agent": "Mozilla/5.0 (Macintosh;Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
    "X-Amzn-Trace-Id": "Root=1-5e351b64-6460c17269bd0256d3060679"
  },
  "json": null,
  "origin": "183.1.89.124",
  "url": "http://httpbin.org/post"
}
```

- urllib.request模块中的BaseHandler类
  - HTTPDefaultErrorHandler:用于处理HTTP响应错误/抛出HTTPError类型的异常
  - HTTPRedirectHandler:用于处理重定向
  - HTTPCookieProcessor:用于Cookies
  - ProxyHandler：用于设置代理
  - HTTPPasswordMgr:用于管理密码
  - HTTPBasicAuthHandler：用于管理认证
<br/>
<br/>
<br/>
- 验证
  - 遇到一些网站弹出提示框验证
  - HTTPBasicAuthHandler

```python
from urllib.request import HTTPPasswordMgrWithDefaultRealm,HTTPBasicAuthHandler,build_opener
from urllib.error import URLError

username 'username'
password = 'password'
url = 'xxxxx'

p = HTTPPasswordMgrWithDefaultRealm()
p.add_password(None,url,username,password)
auth_handler = HTTPBasicAuthHandler(p)
opener = build_opener(auth_handler)

try:
  result = opener.open(url)
  html = result.read().decode('utf-8')
  print(html)
except URLError as e:
  print(e.reason)
```
<br/>
<br/>


- 代理
  - 在本地搭建代理/使用ProxyHandler/参数是字典/键名是协议/键值是代理连接

```python
from urllib.error import URLError
from urllib.request import ProxyHandler,build_opener

proxy_handler = ProxyHandler({
  'http':'http://127.0.0.1:9743',
  'https':'https://127.0.0.1:9743'
})
opener = build_opener(proxy_handler)
try:
  response = opener.open('https://www.zhihu.com')
  print(response.read().decode('utf-8'))
except URLError as e:
  print(e.reason)
```
<br/>
<br/>

- Cookies
  - 申明一个cookiejar对象/利用HTTPCookieProcessor构建Handler

```python
# %%
import http.cookiejar,urllib.request

cookie = http.cookiejar.CookieJar()
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('http://www.baidu.com')
for item in cookie:
  print(item.name+'='+item.value)
# %%
```

```
BAIDUID=3469C8B48BE6647ACE75A671537DBD5D:FG=1
BIDUPSID=3469C8B48BE6647A59A6594DD6B7D762
H_PS_PSSID=1466_21115_30489_26350_30501
PSTM=1580540394
delPer=0
BDSVRTM=0
BD_HOME=0
```

- 格式化输出成文件
```python
filename='cookies.txt'
cookies = http.cookiejar.MozillaCookieJar(filename)
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('http://www.baidu.com')
cookie.save(ignore_discard=True,ignore_expires=True)
```
