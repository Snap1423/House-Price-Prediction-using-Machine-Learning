import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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
# model = LinearRegression()
# model = DecisionTreeRegressor(random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)

# make model predictions
predictions = model.predict(X_test)
print("The predictions made by the model: \n", predictions[:5])

# Evaluate RMSE
eval = np.sqrt(mean_squared_error(y_test,predictions))
print("RMSE: ", eval)

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





