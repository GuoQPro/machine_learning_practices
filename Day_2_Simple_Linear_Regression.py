import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Datasets/studentscores.csv')

# make sure X is a (n, m) array. 
# because we have only 1 feature, we need to reshape it to (n, 1)
X = np.array(dataset.Hours).reshape(-1, 1) 

Y = dataset.Scores



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0) 


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# train model
regressor = regressor.fit(X_train, Y_train)

# predict using trained model
Y_pred = regressor.predict(X_test)

# mesure
print('Coefficients: ', regressor.coef_)

# The mean squared error
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))

# visualization
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Y_pred, color = 'blue')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

