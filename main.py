import numpy as np
import pandas as pd

dataset = pd.read_csv("diabetes.csv")

# dataset.shape
 
#getting stats of data
# print(dataset.describe)

# 0 ----> non-diabetic
# 1 ----> Diabetic

# print(dataset.groupby('Outcome').mean())

X = dataset.iloc[:,:-1].values
# print(X)
Y = dataset.iloc[:,-1].values
# print(Y)

#split the dataset into train and test
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X , Y, test_size = 0.25,random_state=1)


#data standardization 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

#training the model 
#Here we have classification problem so we will use the SVM algori
from sklearn.svm import SVC
classifier = SVC( kernel = 'rbf' , random_state = 0)

classifier.fit(X_train , Y_train)
Y_pred =(classifier.predict(X_test))
YPred = classifier

print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1)) #concatenating predicted and actual values


#acuuracy score

from sklearn.metrics import accuracy_score , confusion_matrix
ac = accuracy_score(Y_test, Y_pred)
print(ac)