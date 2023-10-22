import pandas as pd
from sklearn import linear_model

dataframe = pd.read_csv("datasets/data.csv")
X = dataframe[['Volume', 'Weight']]
y = dataframe["CO2"]

model = linear_model.LinearRegression()
model.fit(X.values, y.values)
print(f"Predicted CO2: {model.predict([[1600, 1150]])}")
print(model.coef_[0] * 1600 + model.coef_[1] * 1150 + model.intercept_)
print(model.coef_)
model.score(X.values, y.values)
print(f"intercept: {model.intercept_:0.2f}")
