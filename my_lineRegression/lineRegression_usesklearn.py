import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
x = np.linspace(0, 6, 11) + np.random.randn(11)
x = np.sort(x)
y = x ** 2 + 2 + np.random.randn(11)
x=x[:,np.newaxis]
print(x)
regr = linear_model.LinearRegression()
regr.fit(x,y)
y_pre = regr.predict(x)

plt.scatter(x,y)
plt.plot(x,y_pre)
plt.show()