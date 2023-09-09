import pandas as pd
from functions import *
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy import optimize
np.seterr(all='raise')

#Import data
trainData = pd.read_csv('HousingPrices/train.csv')
testData = pd.read_csv('HousingPrices/test.csv')


#Clean/Engineer Data
X, Xtest, y = cleanData(trainData, testData)

#Split Data for testing purposes
X_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)
del x_, y_

### Manual Linear Regression (Very inefficient not accurate at all, does not like the large dataset)
### Scores ~25,000 on Kaggle Housing Prices Compeition
m = y_train.size
X_train_manual = np.concatenate([np.ones((m,1)), X_train], axis=1)

alpha_manual = 0.0000001
num_iters_manual = 500000

theta = np.zeros(X_train_manual.shape[1])
theta = gradientDescentManual(X_train_manual, y_train, theta, alpha_manual, num_iters_manual)
Xtest_manual = np.concatenate([np.ones((Xtest.shape[0],1)), Xtest], axis=1)
preds = np.dot(Xtest_manual, theta)

### Sklearn Linear Regression Model, much improved from my manual model and works very quickly
### Scores ~ 19,300 on Kaggle
LinearModel = LinearRegression()
LinearModel.fit(X_train, y_train)
preds = LinearModel.predict(Xtest)

print(LinearModel.score(x_cv, y_cv))

### XGB Boosted Regression model is by far the best of the 3, quickly and accurately
### Scores ~ 15,400 on kaggle, top 4000 out of 72k (august 15, 2023) 
BoostedRegression = xgb.XGBRegressor(min_child_weight = 5, eta=0.1, max_depth = 4, reg_lambda = 2)
BoostedRegression.fit(X_train, y_train)
preds = BoostedRegression.predict(Xtest)


# submission = pd.DataFrame({'Id': testData.Id, 'SalePrice': preds})
# submission.to_csv('HousingPrices/Linearsubmission.csv', index = False)

