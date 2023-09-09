from functions import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy import optimize

#Load Data
trainData = pd.read_csv('Titanic/train.csv')
testData = pd.read_csv('Titanic/test.csv')

#Organize/Engineer the data
X, Xtest, y = cleanData(trainData, testData)

#Split Data into test sets
X_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

### Manual Linear Regression model, using a manual cost function and a minimizing algorith for gradient descent
### scores ~0.768 on Kaggle Titanic Comp
m = y_train.size
X_manual = np.concatenate([np.ones((m,1)), X_train], axis=1)
initial_params = np.zeros(X_manual.shape[1])
result = optimize.minimize(costFunction, initial_params, (X_manual,y_train),jac=True, method='TNC', options={'maxfun': 10000})
preds = manualPrediction(Xtest, result.x)

### Sklearn Logistic Model
### Scores ~0.76 on Kaggle Titanic Comp
LogisticModel = LogisticRegression()
LogisticModel.fit(X_train, y_train)
print(LogisticModel.score(X_train, y_train))
print(LogisticModel.score(x_cv, y_cv))
preds = LogisticModel.predict(Xtest)

### XGB Classifier model, very efficient and the most accurate of the 3
### Scores 0.787 on Kaggle Titanic Comp, ~1500/15,000
boostedReg = xgb.XGBClassifier(eta=0.01, reg_lambda = 0.8, subsample = 0.8, max_depth = 8, min_child_weight = 6, reg_alpha = 0.1)
boostedReg.fit(X_train, y_train)
print(boostedReg.score(X_train, y_train))
print(boostedReg.score(x_cv, y_cv))
preds = boostedReg.predict(Xtest)

#Generate Submission
submission = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': preds})
submission.to_csv('Titanic/XGBSub.csv', index=False)