# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', None)

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sub_data = pd.read_csv("../input/gender_submission.csv")

#train_data.describe()
#train_data.info()

# clean data

datasets = [train_data, test_data] # list object

# I do think the Cabin number is critical, but unfortunately we have too few
# So We have to drop it.


train_data.drop(labels = ['Cabin'], axis = 1, inplace = True)
test_data.drop(labels = ['Cabin'], axis = 1, inplace = True)
train_data.Embarked.fillna('S', inplace=True)
test_data.Embarked.fillna('S', inplace=True)
    
#sns.distplot(train_data.Age[train_data.Age.isnull() == False])

train_data.Age.fillna(train_data.Age.median(),inplace = True)
test_data.Age.fillna(test_data.Age.median(),inplace = True)

test_data.Fare.fillna(test_data.Fare.median(), inplace = True)

#train.Embarked.value_counts()
#train.Embarked.isnull().sum()
#train.Embarked[train.Embarked.isnull()]


# Feature engineering
for dataset in datasets:
    dataset.loc[dataset.Sex == "male", 'Sex'] = 1
    dataset.loc[dataset.Sex == "female", 'Sex'] = 2
    
    dataset.loc[dataset.Embarked == "S", "Embarked"] = 1
    dataset.loc[dataset.Embarked == "C", "Embarked"] = 2
    dataset.loc[dataset.Embarked == "Q", "Embarked"] = 3


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

all_Embarked = train_data.Embarked.append(test_data.Embarked)
embark_array = np.array(all_Embarked).reshape(-1, 1)
enc.fit(embark_array)

sparse_embark_array = enc.transform(np.array(train_data.Embarked).reshape(-1, 1)).toarray()
for i in range(1, sparse_embark_array.shape[1]):
    kword = "embarkT%d" % i
    temp_df = pd.DataFrame({kword:sparse_embark_array[:, i]})
    train_data = train_data.join(temp_df)
        
train_data.drop(labels=["Embarked"], axis=1, inplace=True)

sparse_embark_array = enc.transform(np.array(test_data.Embarked).reshape(-1, 1)).toarray()
for i in range(1, sparse_embark_array.shape[1]):
    kword = "embarkT%d" % i
    temp_df = pd.DataFrame({kword:sparse_embark_array[:, i]})
    test_data = test_data.join(temp_df)
        
test_data.drop(labels=["Embarked"], axis=1, inplace=True)

# Treat title

train_data["Title"] = train_data.Name.str.extract("([A-Za-z]+)\.", expand=True)
test_data["Title"] = test_data.Name.str.extract("([A-Za-z]+)\.", expand=True)
    
all_Title = train_data.Title.append(test_data.Title)
enc = OneHotEncoder(handle_unknown='ignore')
title_array = np.array(all_Title).reshape(-1, 1)
enc.fit(title_array)

sparse_title_array = enc.transform(np.array(train_data.Title).reshape(-1, 1)).toarray()
for i in range(1, sparse_title_array.shape[1]):
    kword = "Title%d" % i
    temp_df = pd.DataFrame({kword:sparse_title_array[:, i]})
    train_data = train_data.join(temp_df)
    
sparse_title_array = enc.transform(np.array(test_data.Title).reshape(-1, 1)).toarray()
for i in range(1, sparse_title_array.shape[1]):
    kword = "Title%d" % i
    temp_df = pd.DataFrame({kword:sparse_title_array[:, i]})
    test_data = test_data.join(temp_df)

train_data.drop(labels = ["Name", "Ticket", "PassengerId", "Title"], axis = 1, inplace = True)
test_data.drop(labels = ["Name", "Ticket", "PassengerId", "Title"], axis = 1, inplace = True)

train_y = train_data.Survived     
train_data.drop(labels=['Survived'], axis = 1, inplace = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
datasets = [train_data, test_data] # list object
for dataset in datasets: 
    ages_array = np.array(dataset["Age"]).reshape(-1, 1) 
    fares_array = np.array(dataset["Fare"]).reshape(-1, 1) 
    dataset["Age"] = scaler.fit_transform(ages_array)
    dataset["Fare"] = scaler.fit_transform(fares_array)

# create validation dataset
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(train_data, train_y, test_size=0.33)

## Start to choose algorithm
# logistic regression/svm/knn

from sklearn.metrics import mean_squared_error, r2_score

# logistic regression
from sklearn.linear_model import LogisticRegression
logisticClassfier = LogisticRegression(solver='liblinear').fit(X_train, y_train)
logisticClassfier.score(X_cv, y_cv)
y_neigh = logisticClassfier.predict(test_data)
#test_y = logisticClassfier.predict(test_data)


# svm
#from sklearn.svm import SVC
#for x in range(1, 11):
#    c_value = 0.1 * x;
#    clf = SVC(C=c_value, kernel="linear");
#    clf.fit(X_train, y_train);
#    print("Score with C %.2f is %f" % (c_value, clf.score(X_cv, y_cv)))

#knn
#from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=2)
#neigh.fit(X_train, y_train)
#neigh.score(X_cv, y_cv)
#y_neigh = neigh.predict(test_data)

#from keras.layers import Dense
#from keras.models import Sequential
## Initialising the NN
#model = Sequential()
##149-20-20-20-20-20-20-20-20-1
## layers
#model.add(Dense(units = 149, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#
## Compiling the ANN,
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
## Train the ANN
#model.fit(X_train, y_train, batch_size = 32, epochs = 1000,validation_data=(X_cv,y_cv))
#y_neigh = model.predict(test_data)
#y_neigh = (y_neigh > 0.5).astype(int).reshape(test_data.shape[0])

for x in range(1, len(y_neigh)):
    sub_data.iloc[x, 1] = y_neigh[x]

sub_data.to_csv('submission.csv', index=False)






















