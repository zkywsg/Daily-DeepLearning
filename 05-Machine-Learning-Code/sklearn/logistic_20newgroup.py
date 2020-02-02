# coding:utf-8
import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore',category=ConvergenceWarning,module='sklearn')
t0 = timeit.default_timer()

# we use SAGA solver
solver = 'saga'

# Tunrn down for fater run time
n_samples = 1000

X,y = fetch_20newsgroups_vectorized('all',return_X_y=True)
X = X[:n_samples]
y = y[:n_samples]
# print(X.shape,y.shape) - (1000,130107) / (1000,)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y,test_size=0.1)

train_samples,n_features = X_train.shape
# print(train_samples) - 900
# print(n_features) - 130107
n_classes = np.unique(y).shape[0]
# print(n_classes) - 20

models = {'ovr':{'name':'one versus Rest', 'iters':[1,2,4]},
          'multinomial':{'name':'Multinomial','iters':[1,3,7]}}

for model in models:
    # add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]
    times = [0]
    densities = [1]

    model_params = models[model]

    # Small number of epochs for fast runtime
    for this_max_iter in model_params['iters']:
        print('[model=%s ,solver=%s] Number of epochs: %s'%
              (model_params['name'], solver,this_max_iter))
        lr = LogisticRegression(solver=solver,
                                multi_class=model,
                                penalty='l1',
                                max_iter=this_max_iter,
                                random_state=42,)
        t1 = timeit.default_timer()
        lr.fit(X_train,y_train)
        train_time = timeit.default_timer() - t1

        y_pred = lr.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        density = np.mean(lr.coef_ != 0,axis=1)*100
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    models[model]['times'] = times
    models[model]['desities'] = densities
    models[model]['accuracies'] = accuracies
    print('Test accuracy for model %s: %s.4f'%(model,accuracies[-1]))
    print('%% non-zero coefficients for model %s,'
          'per class:\n%s'%(model,densities[-1]))
    print('Run time(%i epochs) for model %s:'
          '%.2f' %(model_params['iters'][-1],model,times[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)

for model in models:
    name = models[model]['name']
    times = models[model]['times']
    accuracies = models[model]['accuracies']
    ax.plot(times,accuracies,marker='o',
            label='Model:%s'%name)
    ax.set_xlabel('Train time(s)')
    ax.set_ylabel('Test accuracy')
ax.legend()
fig.suptitle('Multinomial vs One-vs-Rest Logistic L1\n'
             'Dataset %s'%'20newsfroups')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
run_time = timeit.default_timer()-t0
print('Example run in %.3f s'%run_time)
plt.show()
