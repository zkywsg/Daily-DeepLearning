from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)
# [[0.5        0.         1.        ]
#  [1.         0.5        0.33333333]
#  [0.         1.         0.        ]]

X_test = np.array([[-3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print(X_test_minmax)
# [[0.5        0.         1.        ]
#  [1.         0.5        0.33333333]
#  [0.         1.         0.        ]]
# [[-1.5         0.          1.66666667]]

print(min_max_scaler.scale_)
# [0.5        0.5        0.33333333]

print(min_max_scaler.min_)
# [0.         0.5        0.33333333]

