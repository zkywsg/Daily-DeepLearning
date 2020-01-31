### Linear Regression 线性回归例子

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

# 加载diabetes数据集
diabetes_X,diabetes_y = datasets.load_diabetes(return_X_y=True)

# 只用一个特征
diabetes_X = diabetes_X[:,np.newaxis,2]

# 分割训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 分割训练集和测试集的标签
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 线性回归对象
regr = linear_model.LinearRegression()

# 用训练集训练模型
regr.fit(diabetes_X_train,diabetes_y_train)

# 在测试集上预测
diabetes_y_pred = regr.predict(diabetes_X_test)

# 系数
print('Coefficients:\n',regr.coef_)
# 均方误差
print('Mean squared error:%.2f'%mean_squared_error(diabetes_y_test,diabetes_y_pred))
# 系数的误差 /1是最好的
print('Coefficient of determination:%.2f'%r2_score(diabetes_y_test,diabetes_y_pred))

# %%
# 画图
plt.scatter(diabetes_X_test,diabetes_y_test,color='black')
plt.plot(diabetes_X_test,diabetes_y_pred,color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
# %%
```
\>>>
<br/>
Coefficients:[938.23786125]
<br/>
Mean squared error:2548.07
<br/>
Coefficient of determination:0.47

![](1.png)
