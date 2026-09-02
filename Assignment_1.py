import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
import seaborn as sn
import matplotlib.pyplot as plt


url= 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_2_regression_model/Exercise2BikeSharing.csv'

df = pd.read_csv(url)

df.head(10)

df.info()


df = df.drop(
    ['dteday', 'instant', 'casual', 'registered'],
    axis=1
)  # Drop the date as it is an incompatible datatype, and drop instant,
   # casual and registered as they are part of cnt.

corr_matrix_bike = df.corr()  # Creates a correlation matrix

corr_matrix_bike['cnt'].sort_values(
    ascending=False
)  # Focuses on the correlation between the remaining columns and cnt.

sn.heatmap(corr_matrix_bike, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

x = df.drop(
    'cnt',
    axis=1
)  # Dropping count as we want this as our predictor

y = df['cnt']


# Split into train and test data sets.
X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)  # We use an 80/20 split. Common practice in train/test split data.


# Training.
reg_bike = LinearRegression()

reg_bike.fit(
    X_train,
    Y_train
)

y_pred = reg_bike.predict(
    X_test
)


# Calculate MSE, MAE, etc.
mse = mean_squared_error(
    Y_test,
    y_pred
)

mae = mean_absolute_error(
    Y_test,
    y_pred
)

rmse = root_mean_squared_error(
    Y_test,
    y_pred
)

r2 = r2_score(
    Y_test,
    y_pred
)


# Print results.
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")


plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    Y_test,
    y_pred,
    alpha=0.5
)


plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (LR)")

plt.plot(
    [min(Y_test), max(Y_test)],
    [min(Y_test), max(Y_test)],
    linestyle='--',
    color='red',
    lw=2
)

plt.show()


# The results are not very good.
# Therefore we will move on to a more advanced model,
# like an SVM and XGBoost.

# Defining and training the XGBoost model.
X, y = x, y

dtrain = xgb.DMatrix(
    X_train,
    label=Y_train
)

dtest = xgb.DMatrix(
    X_test,
    label=Y_test
)


# Defining the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Parameters that GridSearchCV will test
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2, 0.3]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1
)

# Test all parameter combinations using the TRAINING data
grid_search.fit(
    X_train,
    Y_train
)

# Show the best parameters
print("Best XGBoost Parameters:")
print(grid_search.best_params_)

# Best model found by GridSearchCV
vibe = grid_search.best_estimator_

# Predict using the untouched test dataset
y_pred_xgb = vibe.predict(
    X_test
)

vibe.save_model(
    'xgboost_model_bike.model'
)


# Calculation of evaluation metrics.
xgb_mse = mean_squared_error(
    Y_test,
    y_pred_xgb
)

xgb_mae = mean_absolute_error(
    Y_test,
    y_pred_xgb
)

xgb_rmse = root_mean_squared_error(
    Y_test,
    y_pred_xgb
)

xgb_r2 = r2_score(
    Y_test,
    y_pred_xgb
)


# Print the results.
print(f"XGBoost Mean Squared Error: {xgb_mse}")
print(f"XGBoost Mean Absolute Error: {xgb_mae}")
print(f"XGBoost Root Mean Squared Error: {xgb_rmse}")
print(f"XGBoost R2 Score: {xgb_r2}")


xgb.plot_importance(
    vibe
)  # Shows the attributes that impact the model score the most.

plt.show()


plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    Y_test,
    y_pred_xgb,
    alpha=0.5
)


plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (XGB)")

plt.plot(
    [min(Y_test), max(Y_test)],
    [min(Y_test), max(Y_test)],
    linestyle='--',
    color='red',
    lw=2
)

plt.show()