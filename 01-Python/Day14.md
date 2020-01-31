### 1.协程
- 又叫做微线程/Coroutine
- 子程序或者叫函数/在所有语言中都是层级调用
- 子程序调用总是一个入口/一次返回/调用顺序是明确的/协程不一样
- 协程看上去也是子程序/但是执行过程中/在子程序内部可以终端/去执行别的子程序

```python
def A():
    print('1')
    print('2')
    print('3')

def B():
    print('x')
    print('y')
    print('z')
```

- 假设由协程执行/在执行A的过程/可以随时中断/去执行B
<br/>
\>>> 1 2 x y 3 z
<br/>
<br/>
- A/B的执行看起来像多线程/但协程的特点在于是一个线程执行
- 最大优势/执行效率高/因为子程序切换不是线程切换/而是程序自身控制/没有线程切换的开销/
- 第二大又是是不需要多线程的锁机制/因为只有一个线程/不存在同时写冲突/只需要判断状态
- 因为协程是一个线程执行/在多核cpu/多进程+协程/充分利用多核/有发挥协程的高效率
<br/>
<br/>
```python
# 对协程的支持通过generator实现
# 传统的生产者-消费者模型是一个线程写消息/一个取消息/通过锁控制队列和等待/容易死锁
# 用协程/生产者生产消息后/通过yield跳转到消费者开始执行/待消费者执行完毕/切换回生产者继续生产
def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER]Consuming %s...'%n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER]Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER]Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)
```

```
[PRODUCER]Producing 1...
[CONSUMER]Consuming 1...
[PRODUCER]Consumer return: 200 OK
[PRODUCER]Producing 2...
[CONSUMER]Consuming 2...
[PRODUCER]Consumer return: 200 OK
[PRODUCER]Producing 3...
[CONSUMER]Consuming 3...
[PRODUCER]Consumer return: 200 OK
[PRODUCER]Producing 4...
[CONSUMER]Consuming 4...
[PRODUCER]Consumer return: 200 OK
[PRODUCER]Producing 5...
[CONSUMER]Consuming 5...
[PRODUCER]Consumer return: 200 OK
```
- consumer函数是一个generator/把一个consumer传入produce
- 首先调用c.send(None)启动生成器
- 一旦产生东西/通过c.send(n)切换到consumer执行
- consumer通过yield拿到消息/处理后/又通过yield把结果传回
- produce拿到consumer处理的结果/继续生产下一条消息
- produce决定不生产了/通过c.close()关闭consumer
- 整个流程没有锁/只有一个线程执行。
- 子程序就是协程的一种特例

### 2.asyncio

- asyncio直接内置了对异步IO的支持
- asyncio的编程模型是一个消息循环/从asyncio模块直接获取一个EventLoop的引用/然后把需要执行的协程扔到EventLoop中/实现异步IO

```python
import asyncio
# %%
@asyncio.coroutine
def hello():
    print('Hello world')
    # 异步调用asyncio.sleep(1)
    r = yield from asyncio.sleep(1)
    print('Hello again!')

# 获取EventLoop
loop = asyncio.get_event_loop()
# 执行coroutine
loop.run_until_complete(hello())
loop.close()
# %%
```

- @asyncio.coroutine把一个generator标记为coroutine类型/然后把这个coroutine扔到EventLoop执行
- hello()会首先打印处**hello world**/yield from愈发可以让我们方便调用另一个generator/由于asyncio.sleep()也是一个coroutine/所以线程不会等待asyncio.sleep/直接中断并且执行下一个消息循环
- 当asyncio.sleep()返回/线程可以从yield from拿到返回值/此处是None/然后接着执行下一个语句
- 把asyncio.sleep(1)看成是一个耗时1秒的IO操作/在此期间/祝线程并没有等待/而是去执行EventLoop中其他可以执行的coroutine/因此可以实现并发执行

```python
# 用Task封装两个coroutine
# %%
import threading
import asyncio

@asyncio.coroutine
def hello():
    print('Hello world!(%s)'%threading.currentThread())
    yield from asyncio.sleep(1)
    print('Hello again!(%s)'%threading.currentThread())

loop = asyncio.get_event_loop()
tasks = [hello(),hello()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
# %%
```

```
Hello world!(<_MainThread(MainThread, started 4587290048)>)
Hello world!(<_MainThread(MainThread, started 4587290048)>)
(1秒)
Hello again!(<_MainThread(MainThread, started 4587290048)>)
Hello again!(<_MainThread(MainThread, started 4587290048)>)
```

- 打印的当前线程名称可以看出/两个coroutine是同一个线程并发执行的
- 如果把asyncio.sleep(1)换成真正的IO操作/则多个coroutine就可以由一个线程并发执行
- asyncio提供了完善的异步IO支持
- 异步操作需要在coroutine中通过yield from 完成
- 多个coroutine可以封装成一组Task然后并发执行

### async/await
- asyncio提供的@asyncio.coroutine可以把一个generator标记为coroutine类型/然后在coroutine内部用yield from调用另一个coroutine实现一步操作
- 为了简化并更好地标识异步IO/async和awit可以让coroutine的代码更简洁易读
- async和awit是针对coroutine的新语法
    - 把@asyncio.coroutine替换成async
    - 把yield from替换成await

```python
@asyncio.coroutine
def hello():
    print('Hello world!')
    r = yield from asyncio.sleep(1)
    print('Hello again!')
```

```python
# 新语法重新编写后
async def hello():
    print('Hello world')
    r = await asyncio.sleep(1)
    print('Hello again!')
```


### aiohttp

- asyncio可以实现单线程并发IO操作/ 如果在客户端/发挥的威力不大
- 如果把asyncio用在服务器端/web服务器/由于HTTP连接就是IO操作/因此可以用单线程+coroutine实现多用户的高并发
- aiohttp是给予asyncio实现的HTTP框架
