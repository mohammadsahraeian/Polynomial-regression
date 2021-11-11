import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_read = pd.read_csv('data.csv')
df = pd.DataFrame(csv_read)
x = np.c_[df['x']]
y = np.c_[df['y']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()



#model.fit(x, y)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_test)
model.fit(x_poly, y_test)
y_pred = model.predict(x_poly)
plt.scatter(x_test, y_pred, c='black')
plt.scatter(x_train, y_train, c='b')
print(model.predict(df['x'])) # اگر بخواهم پیش بینی یک عدد(سلول اکسل) را ب من نشان بده، باید چکار کنم؟
print(x_test, y_test) # چطور باید بنویسم ک خطا نده؟
plt.plot(x, y)
plt.show()