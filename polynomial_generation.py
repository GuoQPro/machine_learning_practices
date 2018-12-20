# generate a polynomial which best match the given data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({"value": [1, 2, 6, 7, 9, 13, 18, 20]})
data["level"] = range(1, len(data.value) + 1)

from sklearn.linear_model import Ridge

regressor = Ridge()


X = np.array(data.level).reshape(-1, 1)
Y = data.value

def constructPolynomial(X, power):
    result = X
    for i in range(2, power + 1):
        result = np.hstack((result, np.power(X, i)))
    return result


optimal = 0

for i in range(1, 20):
    curX = constructPolynomial(X, i)
    regressor = Ridge()
    regressor = regressor.fit(curX, Y)
    curR2Score = r2_score(Y, np.dot(curX, regressor.coef_))
    mean_error = mean_squared_error(Y, np.dot(curX, regressor.coef_))
    if curR2Score > optimal:
        optimal = curR2Score
        selectedPower = i
        coef = regressor.coef_
        selectedX = curX
        selectedReg = regressor
    print("power = {0:d}, and R2 Score is {1:f}, mean_error = {2:f}".format(i, curR2Score, mean_error))


print("The best choice is: {}, and the result is: {}, with r2_score: {}".format(selectedPower, np.dot(selectedX, coef), optimal))


# draw the result
sns.set()

plt.subplot(211)
plt.scatter(X, Y, marker = 'x', color = 'red', label = "real value")
lineX = np.linspace(0, len(data), 1000).reshape(-1, 1)
polynomialLineX = constructPolynomial(lineX, selectedPower)
predY = selectedReg.predict(polynomialLineX)

plt.plot(lineX, predY, color = "b", label = "predicted curve")
plt.title("We can predict the future {}".format(selectedPower))
plt.legend()

plt.subplot(212)
sns.kdeplot(data)

plt.show()
