import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:, 1:-1].values
y = ds.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
y_pred_2 = lin_reg_2.predict(X_poly)

plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.plot(X, y_pred_2, color='green')
plt.title("Linear Regression vs Polynomial Regression")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
