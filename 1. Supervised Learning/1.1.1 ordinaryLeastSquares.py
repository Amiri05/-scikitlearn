# Ordinary Least Squares (OLS) is a statistical method used in linear regression to estimate the relationship 
# between independent and dependent variables. It finds the "best-fitting" line by minimizing the vertical 
# distances between data points and the regression line.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# Loading the data. We only use a single feature. 
# We split the data and target into training and test sets.
X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

# Model. Note: By default an intercept is added, we can control this behavior by setting the fit_intercept parameter.

regressor = LinearRegression().fit(X_train, y_train)

# Model evaluation
y_pred = regressor.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of Determination: {r2_score(y_test, y_pred):.2f}")

# Plotting the results using Matplotlib.pyplot
fig, ax = plt.subplots(ncols=2, figsize=(10,5), sharex=True, sharey=True)

ax[0].scatter(X_train, y_train, label="Train Data Points")
ax[0].plot(X_train, regressor.predict(X_train), linewidth=3, color="tab:orange", label="Model Predictions")
ax[0].set(xlabel="Feature", ylabel="Target", title="Train Set")
ax[0].legend()

ax[1].scatter(X_test, y_test, label="Test data points")
ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
ax[1].legend()

fig.suptitle("Linear Regression")

plt.show()