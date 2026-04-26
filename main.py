import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV    

data = pd.read_csv("housing.csv")
print(data.head())

print(data.info())
print(data.describe())

# removes all the empty rows which are null/any missing value
data = data.dropna()
# removed ocean proximity
data = data.drop("ocean_proximity", axis=1)
# to check values visually
print(data.isnull().sum())

# Split the data into X and Y axis

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]
print(X.head())
print(y.head())

# train and split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train the model
model_1 = LinearRegression()
model_2 = DecisionTreeRegressor(random_state=42)
model_3 = RandomForestRegressor(random_state=42)
model_1.fit(X_train,y_train)
model_2.fit(X_train,y_train)
model_3.fit(X_train,y_train)

# Cross Validation

#linear regression
lin_scores = cross_val_score(model_1, X,y,scoring="neg_mean_squared_error",cv=5)
lin_rmse = np.sqrt(-lin_scores)

#decision tree
tree_scores = cross_val_score(model_2,X,y,scoring="neg_mean_squared_error",cv=5)
tree_rmse = np.sqrt(-tree_scores)

# random forest
rf_scores = cross_val_score(model_3, X, y, scoring="neg_mean_squared_error", cv=5)
rf_rmse = np.sqrt(-rf_scores)


print("Linear rmse: ", lin_rmse.mean())
print("tree rmse: ", tree_rmse.mean())
print("random forest rmse: ", rf_rmse.mean())



# make model predictions
predictions = model_1.predict(X_test)
print("The predictions made by the model: \n", predictions[:5])

# Evaluate RMSE
eval = np.sqrt(mean_squared_error(y_test,predictions))
print("RMSE: ", eval)

# hyperparameter

param_grid = {
    "n_estimators": [50,100],
    "max_depth": [None,5,10]
}

grid = GridSearchCV(model_3,param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train,y_train)
best_model = grid.best_estimator_
print(grid.best_params_)
# graph 1
plt.figure(1)
plt.scatter(y_test, predictions)
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("Random forest: Actual vs predicted")

# error distribution
plt.figure(2)
errors = y_test - predictions
plt.hist(errors, bins=50)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Error distribution")
plt.show()

# Final model graph 
plt.figure(3)
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Final Model: Actual vs Predicted")
plt.show()




