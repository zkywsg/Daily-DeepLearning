```python
import tensorflow as tf

# create a tensor
hello = tf.constant('hello world')
print(hello)
```
\>>>
tf.Tensor(b'hello world', shape=(), dtype=string)


```python
# to access a tensor value , call numpy()
hello.numpy()
```
\>>>b'hello world'
