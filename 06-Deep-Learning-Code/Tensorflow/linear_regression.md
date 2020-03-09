```python
import tensorflow as tf
import numpy as np
rng = np.random

# Parameters.
learning_rate = 0.01
training_step = 1000
display_step = 50


# Training Data.
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = X.shape[0]
```

```python
# Weight and Bias, initialized randomly.
W = tf.Variable(rng.randn(),name='weight')
b = tf.Variable(rng.randn(),name='bias')

# Linear regression
def linear_regression(x):
    return W*x + b

# Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true,2))/(2*n_samples)


# stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)
```


```python
# optimization process
def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred,Y)

    # compute gradients.
    gradients = g.gradient(loss,[W,b])

    # update W and b following gradients.
    optimizer.apply_gradients(zip(gradients,[W,b]))
```

```python
# Run traing for  the given number of steps.
for step in range(1,training_step+1):
    # run the optimization to update W and b values.
    run_optimization()

    if step%display_step==0:
        pred = linear_regression(X)
        loss = mean_square(pred,Y)
        print('step:%i,loss:%f,W:%f,b:%f'%(step,loss,W.numpy(),b.numpy()))
>>>
step:50,loss:0.109837,W:0.148787,b:1.527944
step:100,loss:0.106074,W:0.154847,b:1.484982
step:150,loss:0.102740,W:0.160550,b:1.444552
step:200,loss:0.099788,W:0.165917,b:1.406503
step:250,loss:0.097174,W:0.170967,b:1.370697
step:300,loss:0.094858,W:0.175720,b:1.337000
step:350,loss:0.092808,W:0.180193,b:1.305289
step:400,loss:0.090992,W:0.184403,b:1.275446
step:450,loss:0.089383,W:0.188364,b:1.247362
step:500,loss:0.087959,W:0.192092,b:1.220932
step:550,loss:0.086697,W:0.195601,b:1.196060
step:600,loss:0.085580,W:0.198902,b:1.172652
step:650,loss:0.084591,W:0.202009,b:1.150625
step:700,loss:0.083714,W:0.204933,b:1.129895
step:750,loss:0.082938,W:0.207685,b:1.110387
step:800,loss:0.082251,W:0.210275,b:1.092028
step:850,loss:0.081642,W:0.212712,b:1.074751
step:900,loss:0.081103,W:0.215005,b:1.058492
step:950,loss:0.080626,W:0.217163,b:1.043190
step:1000,loss:0.080203,W:0.219194,b:1.028791
```

```python
import matplotlib.pyplot as plt

# %%
# Graphic display
plt.plot(X,Y,'ro',label='Original data')
plt.plot(X,np.array(W*X+b),label='Fitted line')
plt.legend()
plt.show()
# %%
```
