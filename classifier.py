import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score,recall_score, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# load the data
df = pd.read_csv('HeartAttack.csv')

# separate the classes and the features data
x = df.drop(columns='num')
y = df['num']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Transform features by scaling each feature
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Load the Support Vector Machine Classifier
# from the Grid Search for the best hyperparameters, we get C=4, degree=1, gamma=0.15555555555555556, kernel='poly'
classifier = SVC(kernel='poly', C=4, degree=1, gamma=0.15555555555555556)   
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# Model Report
print("Report:",classification_report(y_test, y_pred))

# Model Accuracy
print("Accuracy:",accuracy_score(y_test, y_pred))

# Model Precision
print("Precision:",precision_score(y_test, y_pred))

# Model Recall
print("Recall:",recall_score(y_test, y_pred))


# Creating a pickle file for the classifier
filename = 'heart-attack-svm-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))