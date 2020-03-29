from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

# [[ 0.         -1.22474487  1.33630621]
#  [ 1.22474487  0.         -0.26726124]
#  [-1.22474487  1.22474487 -1.06904497]]

print(X_scaled.mean(axis=0))
#[0. 0. 0.]

print(X_scaled.std(axis=0))
#[1. 1. 1.]

scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)
#StandardScaler(copy=True, with_mean=True, with_std=True)

print(scaler.mean_)
#[1.         0.         0.33333333]

print(scaler.scale_)
#[0.81649658 0.81649658 1.24721913]

a = scaler.transform(X_train)
print(a.mean(axis=0))
print(a.std(axis=0))
# [0. 0. 0.]
# [1. 1. 1.]

X_test = [[-1., 1., 0.]]
b = scaler.transform(X_test)
print(b.mean(axis=0))
#[-2.44948974  1.22474487 -0.26726124]

