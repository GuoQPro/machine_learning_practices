import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Datasets/50_Startups.csv')

print(dataset.sample())
print(dataset.shape)



# data engineering

# normalize continous value
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset["R&D Spend"] = scaler.fit_transform(np.array(dataset["R&D Spend"]).reshape(-1, 1))
dataset["Administration"] = scaler.fit_transform(np.array(dataset["Administration"]).reshape(-1, 1))
dataset["Marketing Spend"] = scaler.fit_transform(np.array(dataset["Marketing Spend"]).reshape(-1, 1))
dataset["Profit"] = scaler.fit_transform(np.array(dataset["Profit"]).reshape(-1, 1))

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset["State"] = labelencoder.fit_transform(dataset["State"])

plt.subplot(3, 1, 1)
plt1 = sns.heatmap(dataset.corr(), annot = True, cmap = 'RdBu_r')
plt1.set_xticklabels(labels = dataset.columns, rotation = 0)

plt.subplot(3, 2, 3)
plt2 = sns.distplot(dataset.Profit)


plt.subplot(3, 2, 4)
plt3 = sns.barplot(x = "State", y = "Profit", data = dataset)
plt3.set_xticklabels(labels = labelencoder.classes_)


X = dataset.iloc[:, :4]
Y = dataset.Profit

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


plt.subplot(325)
plt.scatter(Y_pred, Y_test, color = 'red') 
plt.plot(Y_test, Y_test, color = 'blue') # dots on this line are predicted perfectly
plt.show()