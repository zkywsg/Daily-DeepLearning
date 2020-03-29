```python
# tensor
import torch
x = torch.empty(5,3)
print(x)
>>>
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])

x = torch.rand(5,3)
print(x)
>>>
tensor([[0.4873, 0.1391, 0.2850],
        [0.5330, 0.5251, 0.5179],
        [0.1078, 0.4236, 0.6415],
        [0.8946, 0.6093, 0.6648],
        [0.5916, 0.5081, 0.5925]])

x = torch.zeros(5,3,dtype=torch.long)
print(x)
>>>
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])

x = torch.tensor([5.5,3])
print(x)
>>>
tensor([5.5000, 3.0000])

x = x.new_ones(5,3,dtype=torch.double)
print(x)
>>>
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)

x = torch.randn_like(x,dtype=torch.float)
print(x)
>>>
tensor([[ 1.0922,  0.0895, -0.1523],
        [-0.4120, -0.6752,  0.2451],
        [-0.7924, -0.8002, -0.9583],
        [ 1.5218,  1.1972, -0.1147],
        [-1.1417, -0.4267,  1.3848]])

print(x.size())
>>>torch.Size([5, 3])
```

```python
# operations

y = torch.rand(5,3)
print(x+y)
>>>
tensor([[ 2.0550,  0.5213, -0.0356],
        [ 0.1630, -0.6193,  1.1509],
        [ 0.1826, -0.2385, -0.3789],
        [ 2.2833,  2.1146,  0.8770],
        [-0.2954, -0.2612,  2.3667]])

print(torch.add(x,y))
>>>
tensor([[ 2.0550,  0.5213, -0.0356],
        [ 0.1630, -0.6193,  1.1509],
        [ 0.1826, -0.2385, -0.3789],
        [ 2.2833,  2.1146,  0.8770],
        [-0.2954, -0.2612,  2.3667]])

res = torch.empty(5,3)
torch.add(x,y,out=res)
print(res)
>>>
tensor([[ 2.0550,  0.5213, -0.0356],
        [ 0.1630, -0.6193,  1.1509],
        [ 0.1826, -0.2385, -0.3789],
        [ 2.2833,  2.1146,  0.8770],
        [-0.2954, -0.2612,  2.3667]])

y.add_(x)
print(y)
>>>
tensor([[ 2.0550,  0.5213, -0.0356],
        [ 0.1630, -0.6193,  1.1509],
        [ 0.1826, -0.2385, -0.3789],
        [ 2.2833,  2.1146,  0.8770],
        [-0.2954, -0.2612,  2.3667]])

# you can use standard Numpy-like indexing with all bells and whistles
print(x[:,1])
>>>
tensor([ 0.0895, -0.6752, -0.8002,  1.1972, -0.4267])

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())
>>>
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

# use .item() to get the value as python number
x = torch.randn(1)
print(x)
print(x.item())
>>>
tensor([0.8289])
0.8288509249687195

# converting a torch tensor to a numpy array
a = torch.ones(5)
print(a)
>>>
tensor([1., 1., 1., 1., 1.])

b = a.numpy()
print(b)
>>>
[1. 1. 1. 1. 1.]

a.add_(1)
>>> tensor([2., 2., 2., 2., 2.])
b
>>>array([2., 2., 2., 2., 2.], dtype=float32)

# see how changing np array changed the the Torch Tensor automatically
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
>>>
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)


# CUDA Tensors
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to('cpu',torch.double))

>>>
tensor([0.7120],device='cuda:0')
tensor([0.7120],dtype=torch.float64)

```
