import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def cleanData(X_train, X_test):
    #Drop passengerid and survived as they are not needed for trianing
    y = X_train['Survived'].to_numpy()
    X_train = X_train.drop(['PassengerId', 'Survived'], axis = 1)

    #Combine data for processing (only for kaggle)
    X_train['train_test'] = 1
    X_test['train_test'] = 0
    Data = pd.concat([X_train,X_test])

    #Split Name data to extract the Title of each person. this is very helpful for age
    Data['Title'] = Data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    Data['Title'] = Data['Title'].apply(lambda title: title if title in common_titles else "Other")

    #Map the average age of each title
    title_mapping = {
        "Master": 5,
        "Miss": 22,
        "Mr": 32,
        "Mrs": 37,
        "Other": 43
    }

    #Find every empty age cell and fill with the average age for their title
    for title, age in title_mapping.items():
        mask = (Data['Age'].isnull()) & (Data['Title'] == title)
        Data.loc[mask, 'Age'] = age

    #Drop the name category in favour of title, drop cabin because of all the empty rows (will work on later)
    #Drop ticket category because it is not very useful
    Data = Data.drop(['Cabin', 'Name', 'Ticket'], axis = 1)
    
    #Split the data back
    X_train = Data[Data.train_test == 1].drop(['train_test'], axis = 1)
    X_test = Data[Data.train_test == 0].drop(['train_test'], axis = 1)
    
    #Fill fare and embarked with the mean and mode respectively
    X_train['Fare'].fillna(X_train['Fare'].mean().round(1), inplace=True)
    X_test['Fare'].fillna(X_test['Fare'].mean().round(1), inplace=True)
    X_train['Embarked'].fillna(X_train['Embarked'].mode(), inplace=True)
    X_test['Embarked'].fillna(X_test['Embarked'].mode(), inplace=True)

    #Combine again to get dummie data
    X_train['train_test'] = 1
    X_test['train_test'] = 0
    Data = pd.concat([X_train,X_test])
    Data = pd.get_dummies(Data)

    X_train = Data[Data.train_test == 1].drop(['train_test'], axis = 1)
    X_test = Data[Data.train_test == 0].drop(['train_test'], axis = 1)

    #Scale the Age and Fare features as the rest is mostly categorical
    scaler = StandardScaler()
    X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
    X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

    X_train = X_train.drop('PassengerId', axis = 1)
    X_test = X_test.drop('PassengerId', axis = 1)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    return X_train, X_test, y

def sigmoid(z):
    z = np.array(z)
    return np.sum(1/(1+np.exp(-z)))

def costFunction(theta, X, y):
    #calculate cost
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)

    #use the cost equation to calculate the total cost
    #this will be used to iptimize numbers for theta
    for i in range(m):
        z_i = np.dot(X[i, :], theta)
        f_wb = sigmoid(z_i)
        J += -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb)
        for j in range(theta.shape[0]):
            grad[j] += (f_wb-y[i])*X[i,j] 
    J = J/m
    grad = grad/m

    return J, grad

def manualPrediction(test, theta):
    #Add row column of 1s to data that will be a placeholder for the intercept parameter
    test = np.concatenate([np.ones((test.shape[0],1)), test], axis = 1)

    m = test.shape[0]
    p = np.zeros(m)
    #Loop thorugh data, using dot fucntion to get prediction and add to 'p' list
    for i in range(m):
        prob = np.dot(test[i,:], theta)
        if prob > 0.5:
            p[i] = 1
        else:
            p[i]= 0

    p = p.astype('int')
    
    return p