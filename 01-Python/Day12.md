### 1. 多进程
- 1.1 多进程的定义
- 1.2 multiprocessing
- 1.3 subprocesses
- 1.4 子进程输入
- 1.5 进程间通信

```python
# 1.1 多进程定义
# 普通的函数调用/调用一次/返回一次/
# fork()调用一次/返回两次/分别在父进程和子进程内返回
# 子进程永远返回0/父进程返回子进程ID/一个父进程可以fork出多个子进程
# 父进程返回子进程ID/子进程只需要调用getppid()可以获得父进程
import os

print('Process (%s) start...' % os.getppid())
# %%
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
# %%
```
\>>>
<br/>
Process (16351) start...
<br/>
I (16351) just created a child process (16378).
<br/>
I am child process (16378) and my parent is 16351.


```python
# 1.2 multiprocessing 支持多平台的多进程模块
from multiprocessing import Process
import os

def run_proc(name):
    print('Run child process %s (%s)' % (name,os.getpid()))

# 创建一个Process的实例/用start()方法/join()方法可以等待子进程结束之后继续往下运行/通常用于进程的同步
if __name__ == '__main__':
    print('Parent process %s.'%os.getpid())
    p = Process(target=run_proc,args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
```
\>>>
<br/>
Parent process 16351.
<br/>
Child process will start.
<br/>
Run child process test (16422)
<br/>
Child process end.

```python
# 1.3 Pool
# 如果需要大量子进程/可以用进程池的方式批量创建子进程
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
```
\>>>
**task0/1/2/3立刻执行/task4要等前面某个执行完/因为pool是默认是4**
<br/>
Parent process 3120.
<br/>
Run task 2 (3129)...
<br/>
Run task 0 (3127)...
<br/>
Run task 1 (3128)...
<br/>
Run task 3 (3130)...
<br/>
Waiting for all subprocesses done...
<br/>
Task 3 runs 0.31 seconds.
<br/>
Run task 4 (3130)...
<br/>
Task 4 runs 0.39 seconds.
<br/>
Task 2 runs 0.82 seconds.
<br/>
Task 1 runs 2.22 seconds.
<br/>
Task 0 runs 2.64 seconds.
<br/>
All subprocesses done.

```python
# p=Pool(5) 可以同时跑5个进程
# Pool默认大小是cpu的核心数/如果你的cpu是8核/那么第九个子进程才会有上面的等待效果
```

```python
# 有时候子进程可能会是一个外部程序/创建子程序/还需要控制子进程的输入和输出
# 1.3 subprocesses 方便启动一个子进程/控制输入和输出
import subprocess

print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)

```
\>>>
<br/>
$ nslookup www.python.org
<br/>
Server:		192.168.31.1
<br/>
Address:	192.168.31.1#53
<br/>
Non-authoritative answer:
<br/>
www.python.org	canonical name = dualstack.python.map.fastly.net.
<br/>
Name:	dualstack.python.map.fastly.net
<br/>
Address: 151.101.228.223
<br/>
Exit code: 0


```python
# 1.4 子进程输入
# 如果子进程需要输入/可以通过communicate()方法输入
import subprocess

print('$ nslookup')
p = subprocess.Popen(['nslookup'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
output,err=p.communicate(b'set q=mx\npython.org\nexit\n')
print(output.decode('utf-8'))
print('Exit code:', p.returncode)
# 相当于在命令行执行nslookup/然后手动输入
set q=mx
python.org
exit
```
\>>>
<br/>
$ nslookup
Server:		192.168.19.4
<br/>
Address:	192.168.19.4#53
<br/>
Non-authoritative answer:
<br/>
python.org	mail exchanger = 50 mail.python.org.
<br/>
Authoritative answers can be found from:
<br/>
mail.python.org	internet address = 82.94.164.166
<br/>
mail.python.org	has AAAA address 2001:888:2000:d::a6
<br/>
Exit code: 0

```python
# 1.5 进程间通信
# multiprocessing模块包装了底层的机制/提供了Queue/Pipes等交流数据的方式
# 父进程创建两个子进程/一个往Queue写数据/一个往Queue读数据
from multiprocessing import Process,Queue
import os,time,random

# %%
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())
# %%
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ == '__main__':
    # 父进程创建Queue/并且传递给各个子进程
    q = Queue()
    pw = Process(target=write,args=(q,))
    pr = Process(target=read,args=(q,))
    pw.start()
    pr.start()
    # 等待pw结束
    pw.join()
    # pr进入死循环/只能强制终止
    pr.terminate()
```
\>>>
<br/>
Process to read: 6466
<br/>
Process to write: 6465
<br/>
Put A to queue...
<br/>
Get A from queue.
<br/>
Put B to queue...
<br/>
Get B from queue.
<br/>
Put C to queue...
<br/>
Get C from queue.


### 2.多线程
- 2.1 threading
- 2.2 Lock
- 2.3 threading.Lock
```python
# 多个任务可以由多进程完成/也可以由一个进程的多个线程完成
# 一个进程至少一个线程
# Python的线程是真正的Posix Thread/不是模拟出来的线程
# python有两个模块：_thread/threading 大多数时候用threading
# 2.1 thread
# 启动一个线程就把一个函数传入并且创建Thread实例/然后调用start()开始执行
import time,threading

# %%
def loop():
    print('thread %s is running...' %threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name,n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)
# %%
print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop,name='LoopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)
# %%
```
\>>>
<br/>
thread MainThread is running...
<br/>
thread LoopThread is running...
<br/>
thread LoopThread >>> 1
<br/>
thread LoopThread >>> 2
<br/>
thread LoopThread >>> 3
<br/>
thread LoopThread >>> 4
<br/>
thread LoopThread >>> 5
<br/>
thread LoopThread ended.
<br/>
thread MainThread ended.

```python
# 2.2 Lock
# 多进程中/同一个变量/各自有一份拷贝存在于每个进程/互相不影响/
# 多线程中/所有变量都是线程共享的/多个线程同时改一个变量是风险极大的
# 改乱的例子
# %%
import time,threading

# 这是你银行的存款
balance = 0

def change_it(n):
    # 先存钱后取钱/结果应该是0
    global balance
    balance = balance+n
    balance = balance-n

def run_thread(n):
    for i in range(1000000):
        change_it(n)

t1 = threading.Thread(target=run_thread,args=(5,))
t2 = threading.Thread(target=run_thread,args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
# 定义了一个共享变量balance/初始值0/启动两个线程/先存后取
# 理论结果是0
# t1/t2交替进行/只要循环次数足够多/balance的结果就不一定是0
# %%
```
\>>>3

```python
# 代码正常执行的时候
初始值 balance = 0

t1: x1 = balance + 5 # x1 = 0 + 5 = 5
t1: balance = x1     # balance = 5
t1: x1 = balance - 5 # x1 = 5 - 5 = 0
t1: balance = x1     # balance = 0

t2: x2 = balance + 8 # x2 = 0 + 8 = 8
t2: balance = x2     # balance = 8
t2: x2 = balance - 8 # x2 = 8 - 8 = 0
t2: balance = x2     # balance = 0

结果 balance = 0
```

```python
# t1/t2 出现的交叉执行时
初始值 balance = 0

t1: x1 = balance + 5  # x1 = 0 + 5 = 5

t2: x2 = balance + 8  # x2 = 0 + 8 = 8
t2: balance = x2      # balance = 8

t1: balance = x1      # balance = 5
t1: x1 = balance - 5  # x1 = 5 - 5 = 0
t1: balance = x1      # balance = 0

t2: x2 = balance - 8  # x2 = 0 - 8 = -8
t2: balance = x2   # balance = -8

结果 balance = -8
```

```python
# 确保balance计算正确/就要给change_it上锁/
# 2.3 threading.Lock

balance = 0
lock = threading.Lock()

def run_thread(n):
    for i in range(100000):
        # 先要获得锁
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()
```


- 启动与CPU核心数量相同的N个线程，在4核CPU上可以监控到CPU占用率仅有102%，也就是仅使用了一核。

- 但是用C、C++或Java来改写相同的死循环，直接可以把全部核心跑满，4核就跑到400%，8核就跑到800%，为什么Python不行呢？

- 因为Python的线程虽然是真正的线程，但解释器执行代码时，有一个GIL锁：Global Interpreter Lock，任何Python线程执行前，必须先获得GIL锁，然后，每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。这个GIL全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核。

- GIL是Python解释器设计的历史遗留问题，通常我们用的解释器是官方实现的CPython，要真正利用多核，除非重写一个不带GIL的解释器。

- 所以，在Python中，可以使用多线程，但不要指望能有效利用多核。如果一定要通过多线程利用多核，那只能通过C扩展来实现，不过这样就失去了Python简单易用的特点。


### 3.ThreadLocal
```python
# 多线程环境/每个线程都有自己的数据/一个线程使用自己局部变量比使用全局变量好/
# 因为局部变量只有线程自己能看见/全局变量要上锁
# 但是局部变量在函数调用和传递的时候很麻烦

def process_student(name):
    std = Student(name)
    do_task_1(std)
    do_task_2(std)

def do_task_1(std):
    do_subtask_1(std)
    do_subtask_2(std)

def do_task_2(std):
    do_subtask_2(std)
    do_subtask_2(std)

# 每个函数一层一层的调用太麻烦/全局变量也不可以/因为每个线程处理的对象不同/不能共享
```

```python
# ThreadLocal
import threading

# 创建全局threadLocal对象
local_school = threading.local()

def process_student():
    # 获得当前线程关联的student
    std = local_school.student
    print('Hello, %s (in %s)' % (std, threading.current_thread().name))

def process_thread(name):
    # 绑定ThreadLocal的student:
    local_school.student = name
    process_student()

t1 = threading.Thread(target=process_thread,args=('Alice',),name='Thread-A')
t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
t1.start()
t2.start()
t1.join()
t2.join()
```
\>>>
<br/>
Hello, Alice (in Thread-A)
<br/>
Hello, Bob (in Thread-B)

- 可以理解全局变量local_school是一个dict/不但可以用local_school.student/还可以绑定其他属性
- ThreadLocal最常用的地方就是为每个线程绑定一个数据库连接/HTTP请求/用户信息等/这样一个线程的所有调用到的处理函数都可以非常方便访问这些资源
- 一个ThreadLocal变量虽然是全局变量/每个线程都只能读写自己线程的独立副本/互不干扰/ThreadLocal解决了参数在一个线程中各个函数之间互相传递的问题
