# Heart Attack Prediction

This project has been done using `Azure App Service` platform on Microsoft Azure. I have deployed a `Web App` to predict possibilities of heart attack in a patient. I have used `Support Vector Machine (SVM)` classification model of `Machine Learning (ML)` to predict the required results. This repository consists of mainly python, html, css files and little bit of javascript to animate the texts inside the html files. Also I have used `Flask` library to intergate the ML model to the web application.

Website link - https://heartattackpredictor.azurewebsites.net/

(It has active service time 60 mins/day only as free F1 service has been used. Currently services have been stopped, so the above website will not work.)

## Dataset
Heart attack or Myocardial Infarction is one of the cardiovascular diseases. ST-segment elevation myocardial infarction (STEMI), the term which is used by the cardiologists, use to describe a classic heart attack. It occurs due to atherosclerosis, restricting blood flow to a wide area of the heart. This leads to continuous damage to the heart muscle due to which the functioning of the heart is completely stopped and may cause death. This attack is severe and needs rapid attention.

In this project, I have used the [Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 

### Description
The dataset has 13 descision parameters with a 'num' column, which represents the output of the dependent parameters. The details of the dataset is taken from this [reference](https://www.ijeat.org/portfolio-item/f30430810621/). The predictable variables are as follows:

1. `Age (age)`: Age of the patient at the time of health checkup
2. `Sex (sex)`: 0 = female and 1 = male
3. `Chest Pain (cp)`: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptotic
4. `Resting Blood Pressure (trestbps)`: Resting blood pressure value of patient in mmHg (unit)
5. `Cholesterol (chol)`: Cholesterol of patient in mg/dl (unit)
6. `Fasting Blood Sugar (fbs)`: 1 = if fbs >120 mg/dl (true), else 0 = if not that (false)
7. `Resting ECG (restecg)`: 0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hypertrophy
8. `Max Heart Rate (thalach)`: Maximum heart rate achieved by any patient
9. `Exercise induced angina (exang)`: 0 = No and 1 = Yes
10. `oldpeak`: Displays the value of ST depression of any patient induced by exercise w.r.t. rest (float values)
11. `slope`: Describes the peak of exercise during ST segment, 0 = up-slope, 1 = flat, 2 = down-slope
12. `No. of major vessels (ca)`: Classified in range 0 to 4 by coloring through fluoroscopy
13. `Thalassemia (thal)`: 1 = normal,2 = fixeddefect, 3 = reversible defect
14. `num`: It's the prediction column for diagnosis of heart attacks. Here, 0 = no possibility of heart attack and 1 = possibilities of heart attack

![Dist](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/AllModelsOutput/PossibilityDistwithAge.png)
![Pairplot](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/AllModelsOutput/Pairplot.png)

## Idea
The main idea of this project is to find out the most accurate machine learning model to predict the presence of heart disease in a patient and deploy the predictor in a web app service.

## Clean the data
Initially I have used the [Cleveland Data](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/processed.cleveland.data) (downloaded from UCI Machine Learning Repository) in the [dataset.py](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/dataset.py) file and clean the dataset by dropping the missing values. After that, I have used the new cleaned dataset [HeartAttack.csv](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/HeartAttack.csv) for the whole project.

## Procedure
Here I have used three platforms `Virtual Studio Code`, `GitHub` and `Azure Web App` to deploy the codes.

### Models
Trained different **Machine Learning** models to get the best accuracy for this dataset.
Accuracies :
- **K-Nearest Neighbour (KNN)** : 83.33 %
- **Decision Tree** : 78.33 %
- **Support Vector Machine (SVM)** : 90 %
- **Random Forest** : 88.33 %
- **XGBoost** : 85 %
- **Artificial Neural Network (ANN)** : 86.67 %

![ANNacc](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/AllModelsOutput/ANNmodelAccuracy.png)
![ANNloss](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/AllModelsOutput/ANNmodelLoss.png)

### Models' Accuracy
![ANNacc](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/AllModelsOutput/ModelAccuracy.png)
![ANNacc](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/AllModelsOutput/AccuracyBarPlot.png)

### LIBSVM
Trained and tested the SVM model with standard [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library also and found accuracy of 86.67 %.

### VS Code
- `classfier.py` : SVM classifier with best hyperparameters is chosen as main model, as it has better accuracy than other models. Loaded the model with train & test data and found 90% accuarcy.
- 
![accuracy](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/project-screenshots/accuracy.png)

- `index.html` : html file of the frontend user interface of the web app. Used some javascripts also to animate some texts in the page.
- `result.html` : html file of the web page where the prediction will be shown. If the output is 0, then no possibility of heart attack and if 1, then there are possibilities of heart attack.
- `styles.css` : css file of the above two html files.
- `app.py` : used Flask library to integrate all the files to the web app.

### GitHub
In the next step, a new repository named `Heart-Attack-Predictor-Web-App` is created in the master branch of GitHub and push all the codes using VS code.

### Azure Web App
After a web app service is created in `Web App` of [Microsoft Azure](https://azure.microsoft.com/en-in/) with resource group, runtime stack as Python 3.9, region etc. Then I deploy all the codes from the GitHub repository in the deployment center.

![azure](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/project-screenshots/azureappservice.png)

## Results
In the result, the web application site is created.

Heart Attack Predictor Web App link - https://heartattackpredictor.azurewebsites.net/

![ui](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/project-screenshots/uipageup.png)

Lets put a random test data
- [x] Age : 50 yrs
- [x] Sex : 1 i.e. male
- [x] Chest Pain : 2 i.e. atypical angina
- [x] Resting Blood Pressure : 140 mmHg
- [x] Cholesterol : 280 mg/dl
- [x] Fasting Blood Sugar : 1 i.e. fbs > 120 mg/dl
- [x] Resting ECG : 0 i.e. normal
- [x] Max Heart Rate : 140
- [x] Exercise induced angina : 0 i.e. No
- [x] ST depression : 1.2 
- [x] ST segment : 1 i.e. flat
- [x] No. of major vessels : 2
- [x] Thalassemia : 1 i.e. normal

![predict](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/project-screenshots/predict.png)

### Output

Got the output **0** i.e. **no possibilities of heart attack**.

![result](https://github.com/Mainak21/Heart-Attack-Predictor-Web-App/blob/master/project-screenshots/noattack.png)

# Reference Papers
1. A Machine Learning Approach for Heart Attack Prediction, Suraj Kumar Gupta; https://www.ijeat.org/portfolio-item/f30430810621/
2. Using Machine Learning Classification Methods to Detect the Presence of Heart Disease, Nestor Pereira; https://www.semanticscholar.org/paper/Using-Machine-Learning-Classification-Methods-to-of-Pereira/6a1468c1075462f256047249634ffb2b1a8f9c1b
3. Prediction and Classification of Heart Disease using AML and Power BI, Debmalya Chatterjee; https://www.semanticscholar.org/paper/Prediction-and-Classification-of-Heart-Disease-AML-Chatterjee-Chandran/4b8752094686b7046fa22978f8e37b1fa5397fb7
4. LIBSVM - a library for SVMs; https://www.csie.ntu.edu.tw/~cjlin/libsvm/
