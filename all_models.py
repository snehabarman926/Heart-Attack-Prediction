import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('HeartAttack.csv')
x = df.drop(columns='num')
y = df['num']

plt.figure(figsize=(14, 5))
sns.distplot(df[df['num'] == 1]['age'], label= "Disease - Yes")
sns.distplot(df[df['num'] == 0]['age'], label= "Disease - No")
plt.legend()
plt.show()

sns.pairplot(df, hue = 'num')
plt.show()

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors=5)
knnClassifier.fit(x_train,y_train)
y_pred_knn=knnClassifier.predict(x_test)
accuracy_knn = accuracy_score(y_pred_knn,y_test)
print("KNN accuracy : "+str(accuracy_knn))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtClassifier = DecisionTreeClassifier(random_state = 42)
dtClassifier.fit(x_train, y_train)
y_pred_dt = dtClassifier.predict(x_test)
accuracy_dt = accuracy_score(y_pred_dt,y_test)
print("Decision Tree Classifier accuracy : "+str(accuracy_dt))

# SVM
from sklearn import svm
from sklearn.svm import SVC
# Load the Support Vector Machine Classifier
# from the Grid Search for the best parameters, we get C=4, degree=1, gamma=0.15555555555555556, kernel='poly'
svmClassifier = SVC(kernel='poly', C=4, degree=1, gamma=0.15555555555555556)   # try with different kernels
svmClassifier.fit(x_train, y_train)
y_pred_svm = svmClassifier.predict(x_test)
print("Report:",classification_report(y_test, y_pred_svm))
print("Precision:",precision_score(y_test, y_pred_svm))
print("Recall:",recall_score(y_test, y_pred_svm))
accuracy_svm = accuracy_score(y_pred_svm, y_test)
print("SVM accuracy : "+str(accuracy_svm))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfClassifier = RandomForestClassifier(random_state = 42)
rfClassifier.fit(x_train, y_train)
y_pred_rf = rfClassifier.predict(x_test)
accuracy_rf = accuracy_score(y_pred_rf,y_test)
print("Random Forest Classifier accuracy : "+str(accuracy_rf))

# XGBoost
from xgboost import XGBClassifier
xgbClassifier = XGBClassifier(objective="binary:logistic", random_state=42)
xgbClassifier.fit(x_train, y_train)
y_pred_xgb = xgbClassifier.predict(x_test)
accuracy_xgb = accuracy_score(y_pred_xgb,y_test)
print("Random Forest Classifier accuracy : "+str(accuracy_xgb))

# Artificial Neural Network
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
#Creating a pipeline
model = Sequential()

#1st hidden layer with input layer
model.add(Dense(units=128,activation="relu",input_dim=13))
model.add(Dropout(0.5))

#2nd hidden layer
model.add(Dense(units=64,activation="relu",))
model.add(Dropout(0.25))

#3rd hidden layer
model.add(Dense(units=128,activation="relu",))
model.add(Dropout(0.5))

#4th hidden layer
model.add(Dense(units=64,activation="relu",input_dim=13))
model.add(Dropout(0.25))

#output layer
model.add(Dense(units=1,activation="sigmoid"))
model.add(Dropout(0.5))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_history=model.fit(x_train,y_train,validation_split=0.2,epochs=1000, batch_size=10,verbose=1)

y_pred_ann = model.predict(x_test)
y_pred_ann = (y_pred_ann > 0.5)
accuracy_ann=accuracy_score(y_pred_ann, y_test)
print("ANN accuracy : "+str(accuracy_ann))

print('Confusion Matrix :', confusion_matrix(y_test, y_pred_ann))
print('Classification Report :', classification_report(y_test,y_pred_ann))

model_history.history.keys()

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('ANN Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Results
accuracy = [accuracy_knn, accuracy_dt, accuracy_svm, accuracy_rf, accuracy_xgb, accuracy_ann]
models = ["K-Nearest Neighbors","Decision Tree","Support Vector Machine","Random Forest","XGBoost", "Artificial Neural Network"]    

for i in range(len(models)):
    print("The achieved accuracy of "+models[i]+" is: "+str(accuracy[i]))

sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(models,accuracy)
plt.xlabel("Models", fontweight="bold")
plt.ylabel("Accuracy", fontweight="bold")
plt.show()

plt.figure(figsize =(15,10))
plt.rcParams["font.size"] = 10

plt.plot(accuracy)
accuracy = list(enumerate(accuracy))
plt.annotate("KNN", accuracy[0])
plt.annotate("Decision Tree", accuracy[1])
plt.annotate("SVM", accuracy[2])
plt.annotate("Random Forest", accuracy[3])
plt.annotate("XGBoost", accuracy[4])
plt.annotate("ANN", accuracy[5])

plt.tick_params(bottom=False, labelbottom=False)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy", fontweight="bold")
plt.show()