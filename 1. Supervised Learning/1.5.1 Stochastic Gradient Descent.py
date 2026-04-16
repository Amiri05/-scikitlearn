# SGD is an iterative optimization algorithm that updates model parameters using only a single or small subset
# of training data points at a time. Unlike traditional gradient descent, which calculates gradients using the 
# entire dataset, SGD enables faster, memory-efficient, and frequent updates, making it ideal for large-scale datasets.

# Load the Diabetes Dataset
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor

# Split the Dataset into Training and Testing Sets
X, y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# SGD Regressor
sgd = SGDRegressor(
    max_iter=100,
    tol=None,
    learning_rate='constant',
    eta0=0.03,
    random_state=2,
    shuffle=True
)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print("SGD R2:", r2_score(y_test, y_pred_sgd))

# Visualizing convergence

import matplotlib.pyplot as plt

reg = SGDRegressor(max_iter=1, warm_start=True, random_state=2)

scores = []

for i in range(100):
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

plt.plot(scores)
plt.xlabel("Iterations")
plt.ylabel("R² Score")
plt.title("SGD Learning Progress")
plt.show()