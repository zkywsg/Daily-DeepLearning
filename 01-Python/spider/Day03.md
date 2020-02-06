### urllib.parse
- 定义了处理URL的标准接口/例如实现URL各部分的抽取/合并和链接转换

- urlparse()
- 实现URL的识别和分段
- 返回了ParseResult对象/包括了scheme/netloc/path/params/query/fragment
```python
from urllib.parse import urlparse

result = urlparse('http://www.baidu.com/index.html;user?id=5#comment')
print(type(result),result)
```
\>>>
<br/>
<class 'urllib.parse.ParseResult'>
<br/>
ParseResult(scheme='http', netloc='www.baidu.com', path='/index.html', params='user', query='id=5', fragment='comment')

- urlparse的参数scheme
  - 加入这个链接没有带协议信息，就会把这个作为默认的协议

```python
from urllib.parse import urlparse

result = urlparse('www.baidu.com/index.html;user?id=5#comment',scheme='https')
print(result)
```
\>>>ParseResult(scheme='https', netloc='', path='www.baidu.com/index.html', params='user', query='id=5', fragment='comment')

- scheme参数只有在url中不包含scheme信息的时候才生效，如果rul中有scheme信息就会返回解析出来的
```python
from urllib.parse import urlparse

result = urlparse('http://www.baidu.com/index.html;user?id=5#comment',scheme='https')
print(result)
```
\>>>ParseResult(scheme='http', netloc='www.baidu.com', path='/index.html', params='user', query='id=5', fragment='comment')



- allow_fragments参数：是否忽略fragment/忽略的话会被解析成path/params/query的一部分

```python
from urllib.parse import urlparse
result = urlparse('http://www.baidu.com/index.html;user?id=5#comment',allow_fragments=False)
print(result)
```
\>>>ParseResult(scheme='http', netloc='www.baidu.com', path='/index.html', params='user', query='id=5#comment', fragment='')

- ParseResult实际是一个元组/可以索引
```python
from urllib.parse import urlparse
result = urlparse('http://www.baidu.com/index.html#comment',allow_fragments=False)
print(result.scheme,result[0],result.netloc,result[1],sep='\n')
```
```
http
http
www.baidu.com
www.baidu.com
```

### urlunparse:接收可迭代对象，长度为6

```python
from urllib.parse import urlunparse

data = ['http','www.baidu.com','index.html','user','a=6','comment']
print(urlunparse(data))
```
```
http://www.baidu.com/index.html;user?a=6#comment
```


### urlsplit:和urlparse相似，不在解析params
```python
from urllib.parse import urlsplit

result = urlsplit('http://www.baidu.com/index.html;user?id=5#comment')
print(result)
````

```
SplitResult(scheme='http', netloc='www.baidu.com', path='/index.html;user', query='id=5', fragment='comment')
```
- SplitResult也是个元组
```python
from urllib.parse import urlsplit

result = urlsplit('http://www.baidu.com/index.html;user?id=5#comment')
print(result.scheme,result[0])
```
\>>>http http


### urlunsplit和urlunparse类似，不过长度只要5
```python
from urllib.parse import urlunsplit

data = ['http','www.baidu.com','index.html','a=6','commment']
print(urlunsplit(data))
```
\>>>http://www.baidu.com/index.html?a=6#commment
### urljoin
- 有base_url/scheme/netloc/path 后三个对base_url进行补充

```python
from urllib.parse import urljoin
print(urljoin('http://www.baidu.com','FAQ.html'))
```
\>>>http://www.baidu.com/FAQ.html


### urlencode:构造GET请求参数
```python
from urllib.parse import urlencode
params = {
    'name':'sb',
    'age':22
}
base_url = 'http://www.baidu.com?'
url = base_url + urlencode(params)
print(url)
```
\>>>http://www.baidu.com?name=sb&age=22

### parse_qs():把GET请求转回字典
```python
from urllib.parse import parse_qs
query = 'name=sb&age=22'
print(parse_qs(query))
```
\>>>{'name': ['sb'], 'age': ['22']}

### parse_qsl():参数变成元组组成的列表
```python
rllib.parse import parse_qsl
query = 'name=sb&age=22'
print(parse_qsl(query))
```
\>>>[('name', 'sb'), ('age', '22')]

### quote():中文参数变成URL编码
```python
from urllib.parse import quote

keyword='测试'
url = 'https://www.baidu.com/s?wd='+quote(keyword)
print(url)
```
```
https://www.baidu.com/s?wd=%E6%B5%8B%E8%AF%95
```

### unquote():还原url编码
```python
from urllib.parse import unquote

url = 'https://www.baidu.com/s?wd=%E6%B5%8B%E8%AF%95'
print(unquote(url))
```
\>>>https://www.baidu.com/s?wd=测试
