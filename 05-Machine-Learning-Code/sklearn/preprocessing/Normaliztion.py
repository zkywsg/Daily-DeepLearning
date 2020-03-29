from sklearn import preprocessing
import numpy as np

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X,norm='l2')
print(X_normalized)
# [[ 0.40824829 -0.40824829  0.81649658]
#  [ 1.          0.          0.        ]
#  [ 0.          0.70710678 -0.70710678]]


normalizer = preprocessing.Normalizer().fit_transform(X)
print(normalizer)
# [[ 0.40824829 -0.40824829  0.81649658]
#  [ 1.          0.          0.        ]
#  [ 0.          0.70710678 -0.70710678]]

