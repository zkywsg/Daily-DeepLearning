```python
import tensorflow as tf
import numpy as np

# mnist dataset parameters.
num_classes = 10
num_features = 784 # 28*28

# training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50
```

```python
# prepare mnist dataset

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# convert to float32.
x_train,x_test = np.array(x_train,np.float32),np.array(x_test,np.float32)
# Flatten images to 1-D vector of 784 features
x_train,x_test = x_train.reshape([-1,num_features]),x_test.reshape([-1,num_features])
# Normalize images value from [0,255] to [0,1]
x_train,x_test = x_train/255.,x_test/255.
```

```python
# use tf.data API to  shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
train_data
>>>
<PrefetchDataset shapes: ((None, 28, 28), (None,)), types: (tf.float32, tf.uint8)>
```

```python
# weight of shape [784,10],the 28*28 image features, and total number of classes.
W = tf.Variable(tf.ones([num_features,num_classes]),name='weight')

# bias of shape [10] the total number of classes
b = tf.Variable(tf.zeros([num_classes]),name='bias')

# Logistic regression
def logistic_regression(x):
    # apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x,W)+b)

# cross-entropy loss function.
def cross_entropy(y_pred,y_true):
    # encode label to a one hot vector.
    y_true = tf.one_hot(y_true,depth=num_classes)

    # clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred,1e-9,1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))

# accuracy metric.
def accuracy(y_pred,y_true):
    # predicted class is the index of highest score in prediction vector
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

optimizer = tf.optimizers.SGD(learning_rate)

```

```python
# optimization process.
def run_optimization(x,y):
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred,y)

    gradients = g.gradient(loss,[W,b])
    optimizer.apply_gradients(zip(gradients,[W,b]))
```

```python
# Run training for the given number of steps.
for step,(batch_x,batch_y) in enumerate(train_data.take(training_steps),1):
    # run the optimization to update W and b values.
    run_optimization(batch_x,batch_y)

    if step%display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred,batch_y)
        acc = accuracy(pred,batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
>>>
step: 50, loss: 225.317780, accuracy: 0.789062
step: 100, loss: 180.205688, accuracy: 0.824219
step: 150, loss: 90.283379, accuracy: 0.906250
step: 200, loss: 136.394806, accuracy: 0.878906
step: 250, loss: 238.000549, accuracy: 0.777344
step: 300, loss: 67.237411, accuracy: 0.933594
step: 350, loss: 61.615898, accuracy: 0.917969
step: 400, loss: 68.632393, accuracy: 0.953125
step: 450, loss: 40.318504, accuracy: 0.957031
step: 500, loss: 94.427498, accuracy: 0.917969
step: 550, loss: 110.136887, accuracy: 0.875000
step: 600, loss: 177.484955, accuracy: 0.824219
step: 650, loss: 75.401894, accuracy: 0.921875
step: 700, loss: 31.143158, accuracy: 0.964844
step: 750, loss: 146.459381, accuracy: 0.859375
step: 800, loss: 75.543419, accuracy: 0.921875
step: 850, loss: 71.618393, accuracy: 0.906250
step: 900, loss: 234.989517, accuracy: 0.835938
step: 950, loss: 43.793526, accuracy: 0.945312
step: 1000, loss: 99.398201, accuracy: 0.902344
```

```python
# Test model on validation set.
pred = logistic_regression(x_test)
print('Test Accuracy:%f'%accuracy(pred,y_test))
>>>
Test Accuracy:0.908500
```

```python
# visualize predictions
# %%
import matplotlib.pyplot as plt
n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(test_images)

# display
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i],[28,28]),cmap='gray')
    plt.show()
    print('Model prediction:%i'%np.argmax(predictions.numpy()[i]))
# %%
```
