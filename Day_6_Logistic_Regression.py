import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Datasets/Social_Network_Ads.csv')


print(dataset.sample())
print(dataset.shape)

# data engineering

# I don't think "User ID" makes any sense
# Let's drop it
dataset.drop(labels = ["User ID"], axis = 1, inplace = True);

# encode catigorical data Gender
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["Gender"] = labelencoder.fit_transform(dataset["Gender"])

# normalize Salary
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset["EstimatedSalary"] = scaler.fit_transform(np.array(dataset["EstimatedSalary"]).reshape(-1, 1))
dataset["Age"] = scaler.fit_transform(np.array(dataset["Age"]).reshape(-1, 1))

# show relationship between each two features
#plt.subplot(3, 1, 1)
#sns.heatmap(dataset.corr(), annot = True, cmap = 'RdBu_r')

# show how gender affectes the probability
#plt.subplot(3, 1, 2)
#plt2 = sns.barplot(x = "Gender", y = "Purchased", data = dataset)
#plt2.set_xticklabels(labels = labelencoder.classes_)

# show how age affectes the probability
#plt.subplot(3, 1, 3)
#sns.stripplot(x = "Purchased", y = "Age", data = dataset, jitter = True)

#plt.show()

X = dataset.iloc[:, :3]
y = dataset.Purchased

# create validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# train model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = "lbfgs")
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)

# show confusion matrix
# TN | FP 
# FN | TP
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("confusion_matrix ", cm)
print("coef_ ", classifier.coef_)

X_test_true = X_test.loc[y_pred == 1]
X_test_false = X_test.loc[y_pred == 0]


# draw 3d 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X_test_true["Gender"], X_test_true["Age"], X_test_true["EstimatedSalary"], c = 'red')
ax.scatter(X_test_false["Gender"], X_test_false["Age"], X_test_false["EstimatedSalary"], c = 'blue')

plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend()
#plt.zlabel('EstimatedSalary')
plt.show()

