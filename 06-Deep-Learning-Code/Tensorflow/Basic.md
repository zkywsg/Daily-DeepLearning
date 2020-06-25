```python
import tensorflow as tf

# define tensor constants
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# various tensor operations
add = tf.add(a,b)
sub = tf.subtract(a,b)
mul = tf.multiply(a,b)
div = tf.divide(a,b)

print('add=',add.numpy())
print('sub=',sub.numpy())
print('mul=',mul.numpy())
print('div=',div.numpy())
```
\>>>
</br>
add= 5
</br>
sub= -1
</br>
mul= 6
</br>
div= 0.6666666666666666


```python
# some more operations
mean = tf.reduce_mean([a,b,c])
sum = tf.reduce_sum([a,b,c])
print('mean=',mean.numpy())
print('sum=',sum.numpy())
```
\>>>
</br>
mean= 3
</br>
sum= 10

```python
# matrix multiplications
matrix1 = tf.constant([[1.,2.],[3.,4.]])
matrix2 = tf.constant([[5.,6.],[7.,8.]])
product = tf.matmul(matrix1,matrix2)
product

>>><tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[19., 22.],
       [43., 50.]], dtype=float32)>

product.numpy()
>>>
array([[19., 22.],
       [43., 50.]], dtype=float32)
```
