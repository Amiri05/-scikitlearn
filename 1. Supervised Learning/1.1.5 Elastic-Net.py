# We already covered this, but ElasticNet combines both L1 (Lasso) and L2 (Ridge) penalties to improve prediction accuracy and handle multicollinearity.
# Elastic-net is useful when there are multiple features that are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

# Loading and Setting up data
sns.get_dataset_names()
tips = sns.load_dataset("tips")
    # print(tips)

tips = pd.get_dummies(tips)
    # print(tips.head(10))

X = tips.drop('tip', axis=1)
y = tips['tip']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=19)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Processing and fitting our data
elastic_net = ElasticNet()
elastic_net.fit(X_train, y_train)
y_pred = elastic_net.predict(X_test)

print("ElasticNet First try")
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Build out a parameter grid.

param_grid = {
    "alpha": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

# Hyperparameter tuning (Note: CV stands for cross-validation)
elastic_cv = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
elastic_cv.fit(X_train, y_train)

y_pred2 = elastic_cv.predict(X_test)

print("After CV and Parameter Tuning")
print(mean_absolute_error(y_test, y_pred2))
print(mean_squared_error(y_test, y_pred2))
print(r2_score(y_test, y_pred2))

# Shows which is the best option
print(elastic_cv.best_estimator_)