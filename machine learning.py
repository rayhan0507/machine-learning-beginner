# jangan langsung salin semua?
# jangan ketik di vs code tapi di google colab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
uploaded = files.upload()

# membaca file csv
df = pd.read_csv(io.BytesIO(uploaded['Salary_Data.csv']))
print(df)

# 
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
print(x, y)

# splitting the data set into training and test set
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3)

# fitting the simple  linear regresion model to  the training dataset
from sklearn.linear_model  import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# prediction  of test and training set result
y_pred = reg.predict(x_test)
x_pred = reg.predict(x_train)

# visualization training result
plt.scatter(x_train, y_train, color = 'green')
plt.plot(x_train, x_pred, color = 'red')
plt.title("salary vs experience(training dataset)")
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

# visualization training result
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_train, x_pred, color = 'red')
plt.title("salary vs experience(training dataset)")
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

# evaluate model
print(np.sqrt(mean_squared_error(y_test, y_pred)))