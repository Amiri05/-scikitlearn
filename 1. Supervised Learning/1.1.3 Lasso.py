# This is a regression analysis method that performs both variable selection and regularization. 
# It improves upon standard linear regression by adding a penalty proportional to the sum of the absolute values 
# of the coefficients (regularization), which forces less significant feature coefficients to zero.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from time import time
from sklearn.linear_model import Lasso
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

import pandas as pd
import seaborn as sns
from matplotlib.colors import SymLogNorm

# We generate a dataset where the number of samples is lower than the total number of features. 
# Regularization introduces a penalty term to the objective function. The target y is a linear combination 
# with alternating signs of sinusoidal signals. Only the 10 lowest out of the 100 frequencies in X are used to 
# generate y, while the rest of the features are not informative. This results in a high dimensional sparse feature 
# space, where some degree of l1-penalization is necessary.

rng = np.random.RandomState(0)
n_samples, n_features, n_informative = 50, 100, 10
time_step = np.linspace(-2, 2, n_samples)
freqs = 2 * np.pi * np.sort(rng.rand(n_features)) / 0.01
X = np.zeros((n_samples, n_features))

for i in range(n_features):
    X[:, i] = np.sin(freqs[i] * time_step)

idx = np.arange(n_features)
true_coef = (-1) ** idx * np.exp(-idx / 10)
true_coef[n_informative:] = 0  # sparsify coef
y = np.dot(X, true_coef)

# A random phase is introduced and some Gaussian noise is added to both the features and the target.
for i in range(n_features):
    X[:, i] = np.sin(freqs[i] * time_step + 2 * (rng.random_sample() - 0.5))
    X[:, i] += 0.2 * rng.normal(0, 1, n_samples)

y += 0.2 * rng.normal(0, 1, n_samples)

# Plot information
plt.plot(time_step, y)
plt.ylabel("target signal")
plt.xlabel("time")
_ = plt.title("Superposition of sinusoidal signals")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
# Testing how long it tasts to run each function and get its r^2 score.
t0 = time()
lasso = Lasso(alpha=0.14).fit(X_train, y_train)
print(f"Lasso fit done in {(time() - t0):.3f}s")

y_pred_lasso = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso r^2 on test data : {r2_score_lasso:.3f}")

# ARD regression is the Bayesian version of the Lasso. 
# It is a suitable option when the signals have Gaussian noise. 
ard = ARDRegression().fit(X_train, y_train)
print(f"ARD fit done in {(time() - t0):.3f}s")

y_pred_ard = ard.predict(X_test)
r2_score_ard = r2_score(y_test, y_pred_ard)
print(f"ARD r^2 on test data : {r2_score_ard:.3f}")

# ElasticNet is the middle ground between Lasso and Ridge, combining l1 and l2 penalty
# The amount of regularization is controlled by the two hyperparameters l1_ratio and alpha.
# l1_ratio = 0 the penalty is pure L2 and the model is equivalent to a Ridge. 
# l1_ratio = 1 is a pure L1 penalty and the model is equivalent to a Lasso.

enet = ElasticNet(alpha=0.08, l1_ratio=0.5).fit(X_train, y_train)
print(f"ElasticNet fit done in {(time() - t0):.3f}s")

y_pred_enet = enet.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(f"ElasticNet r^2 on test data : {r2_score_enet:.3f}")

# Plot and analysis

df = pd.DataFrame(
    {
        "True coefficients": true_coef,
        "Lasso": lasso.coef_,
        "ARDRegression": ard.coef_,
        "ElasticNet": enet.coef_,
    }
)

plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df.T,
    norm=SymLogNorm(linthresh=10e-4, vmin=-1, vmax=1),
    cbar_kws={"label": "coefficients' values"},
    cmap="seismic_r",
)
plt.ylabel("linear model")
plt.xlabel("coefficients")
plt.title(
    f"Models' coefficients\nLasso $R^2$: {r2_score_lasso:.3f}, "
    f"ARD $R^2$: {r2_score_ard:.3f}, "
    f"ElasticNet $R^2$: {r2_score_enet:.3f}"
)
plt.tight_layout()
plt.show()

# Lasso is known to recover sparse data effectively but does not perform well with highly correlated features. 
# If several correlated features contribute to the target, Lasso would end up selecting a single one of them. 
# In the case of sparse yet non-correlated features, a Lasso model would be more suitable.

# ElasticNet introduces some sparsity on the coefficients and shrinks their values to zero. In the presence of 
# correlated features that contribute to the target, the model is still able to reduce their weights without setting them 
# exactly to zero. This results in a less sparse model than a pure Lasso and may capture non-predictive features as well.

# ARDRegression is better when handling Gaussian noise, but is still unable to handle correlated features and requires a larger 
# amount of time due to fitting a prior.