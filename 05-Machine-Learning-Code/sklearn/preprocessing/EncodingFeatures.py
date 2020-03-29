from sklearn import preprocessing
import numpy as np

enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

print(enc.transform([['female', 'from US', 'uses Safari']]))
# [[0. 1. 1.]]

enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray())

# [[1. 0. 0. 1. 0. 1.]
#  [0. 1. 1. 0. 0. 1.]]

print(enc.categories_)
# [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]

genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
print(enc.fit(X))
# OneHotEncoder(categorical_features=None,
#               categories=[['female', 'male'],
#                           ['from Africa', 'from Asia', 'from Europe',
#                            'from US'],
#                           ['uses Chrome', 'uses Firefox', 'uses IE',
#                            'uses Safari']],
#               drop=None, dtype=<class 'numpy.float64'>, handle_unknown='error',
#               n_values=None, sparse=True)

print(enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray())
# [[1. 0. 0. 1. 0. 0. 1. 0. 0. 0.]]

