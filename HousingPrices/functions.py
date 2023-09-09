import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
import pandas as pd
from sklearn_pandas import DataFrameMapper

def cleanData(train, test):
    #SalePrice
    y = train['SalePrice']
    train = train.drop('SalePrice', axis = 1)    

    #Drop useless features that have no correlation or mostly the same values(subject to overfitting)
    train = train.drop(['MoSold','Street', 'Utilities', 'Condition2', 'RoofMatl',
                    'Heating','LowQualFinSF', 'KitchenAbvGr', 'MiscVal','PoolArea'],axis = 1)
    test = test.drop(['MoSold','Street', 'Utilities', 'Condition2', 'RoofMatl', 
                    'Heating','LowQualFinSF', 'KitchenAbvGr', 'MiscVal','PoolArea'],axis = 1)
    
    #Finding and Dropping extreme outliers from training data
    #print(train[train['LotFrontage'] > 250].index) #934, 1298
    #print(train[train['LotArea'] > 200000].index) #313
    #print(train[train['BsmtFinSF1'] > 4000].index) #1298
    #print(train[train['TotalBsmtSF'] > 5000].index) #1298
    train = train.drop([934,1298,313])
    y = y.drop([934,1298,313])
    
    #Combine data to count all null values (only for kaggle)
    train['train_test'] = 1
    test['train_test'] = 0
    data = pd.concat([train,test])
    
    NA_df = pd.DataFrame({'Features': data.isnull().sum()/len(data)}).sort_values(by = 'Features', ascending = False)
    NA_df = NA_df.loc[NA_df['Features']!=0]
    #print(NA_df)

    #Split back into seperate data to avoid data leakage
    X_train = data[data.train_test == 1].drop(['train_test'], axis =1)
    X_test = data[data.train_test == 0].drop(['train_test'], axis =1)

    #These features NA values mean they do not have this feature, fill with 'None'
    X_train[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']] = X_train[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']].fillna('None')
    X_test[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']] = X_test[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']].fillna('None')

    #Fill lot frontage with median value for the neighborhood it is in
    X_train["LotFrontage"] = X_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    X_test["LotFrontage"] = X_test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
    #NA mean no Garage, fill with 'None'
    X_train[['GarageFinish','GarageQual','GarageCond','GarageType']] =X_train[['GarageFinish','GarageQual','GarageCond','GarageType']].fillna('None')
    X_test[['GarageFinish','GarageQual','GarageCond','GarageType']] =X_test[['GarageFinish','GarageQual','GarageCond','GarageType']].fillna('None')
    
    #NA means no basement, fill with 'None
    X_train[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1']] = X_train[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1']].fillna('None')
    X_test[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1']] = X_test[['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1']].fillna('None')

    #NA means none and 0 Area
    X_train['MasVnrType'] = X_train['MasVnrType'].fillna('None')
    X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)

    X_test['MasVnrType'] = X_test['MasVnrType'].fillna('None')
    X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)

    #Use the most common value in MSSubclass to fill MSZoning
    X_train['MSZoning'] = X_train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    X_test['MSZoning'] = X_test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    #NA for functional means 'Typ'
    X_train['Functional'] = X_train['Functional'].fillna('Typ')
    X_test['Functional'] = X_test['Functional'].fillna('Typ')

    #NA likely means it does not exist here, so '0'
    X_train[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2']] = X_train[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2']].fillna(0)
    X_test[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2']] = X_test[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2']].fillna(0)


    #Very few values missing, just fill with the mode
    X_train['KitchenQual'] = X_train.groupby("Neighborhood")["KitchenQual"].transform(lambda x: x.fillna(x.mode()[0]))
    X_train['Electrical'] = X_train.groupby("Neighborhood")["Electrical"].transform(lambda x: x.fillna(x.mode()[0]))
    X_train['SaleType'] = X_train.groupby("Neighborhood")["SaleType"].transform(lambda x: x.fillna(x.mode()[0]))
    X_train['Exterior1st'] = X_train.groupby("Neighborhood")["Exterior1st"].transform(lambda x: x.fillna(x.mode()[0]))
    X_train['Exterior2nd'] = X_train.groupby("Neighborhood")["Exterior2nd"].transform(lambda x: x.fillna(x.mode()[0]))

    X_test['KitchenQual'] = X_test.groupby("Neighborhood")["KitchenQual"].transform(lambda x: x.fillna(x.mode()[0]))
    X_test['Electrical'] = X_test.groupby("Neighborhood")["Electrical"].transform(lambda x: x.fillna(x.mode()[0]))
    X_test['SaleType'] = X_test.groupby("Neighborhood")["SaleType"].transform(lambda x: x.fillna(x.mode()[0]))
    X_test['Exterior1st'] = X_test.groupby("Neighborhood")["Exterior1st"].transform(lambda x: x.fillna(x.mode()[0]))
    X_test['Exterior2nd'] = X_test.groupby("Neighborhood")["Exterior2nd"].transform(lambda x: x.fillna(x.mode()[0]))

    #Likely means no garage, fill with 0
    X_train['GarageYrBlt'] = X_train['GarageYrBlt'].fillna(0)
    X_train['GarageCars'] = X_train['GarageCars'].fillna(0)

    X_test['GarageYrBlt'] = X_test['GarageYrBlt'].fillna(0)
    X_test['GarageCars'] = X_test['GarageCars'].fillna(0)

    #Only 1 house has no basement and garage in the test set
    X_test[['TotalBsmtSF','BsmtUnfSF']] = X_test[['TotalBsmtSF','BsmtUnfSF']].fillna(0)
    X_test['GarageArea'] = X_test['GarageArea'].fillna(0)

    #Check if there are any empty rows
    X_train['train_test'] = 1
    X_test['train_test'] = 0
    data = pd.concat([X_train,X_test])
    NA_df = pd.DataFrame({'Features':data.isnull().sum(axis = 0)/len(data)}).sort_values(by = 'Features',ascending = False)
    NA_df = NA_df.loc[NA_df['Features']!=0]
    #print(NA_df)

    #Convert some simple categorical data to binary/numerical data
    data['CentralAir'] = data['CentralAir'].map({'N':0,'Y':1})
    data['LotShape'] = data['LotShape'].map({'IR3':1,'IR2':2,'IR1':3,'Reg':4})
    data['LandSlope'] = data['LandSlope'].map({'Gtl':1,'Mod':2,'Sev':3})

    QualCondMap = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None':0}
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'KitchenQual', 'HeatingQC', 'FireplaceQu','PoolQC']:
        data[col] = data[col].map(QualCondMap)
    
    data['BsmtExposure'] = data['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})
    data['PavedDrive'] = data['PavedDrive'].map({'N':1,'P':2,'Y':3})
    data['Fence'] = data['Fence'].map({'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'None': 0})


    #convert rest of the categorical data using dummies
    data = pd.get_dummies(data)
    #split back into training and testing
    X_train = data[data.train_test == 1].drop(['train_test'], axis =1)
    X_test = data[data.train_test == 0].drop(['train_test'], axis =1)
    
    #Scale the numerical features, only using float data to avoid the binary categorical data
    numerical_features = data.select_dtypes(include='float').columns.tolist()
    numerical_features.append('LotArea')
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    #Convert to numpy
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    X_train = X_train.astype('float')
    X_test = X_test.astype('float64')
    y = y.to_numpy()

    return X_train, X_test, y

def computeManualCost(X, y, theta):
    #Num of examples
    m = y.shape[0]

    #J = Cost, h = linearEquation 
    J = 0
    h = 0

    for i in range(1,theta.shape[0]):
        h =  h + np.dot(X[:, i], theta[i])
    h = h + theta[0]
    sum = np.sum((-y+h)**2)

    J = 1/(2*m)*sum
    return J

def gradientDescentManual(X, y, theta, alpha, num_iters):
    m = y.shape[0]

    #Learning Coefs
    theta = theta.copy()

    for i in range(num_iters):
        h = 0
        for i in range(1,theta.shape[0]):
            h = h + np.dot(X[:,i], theta[i])
        h += theta[0]
        for i in range(theta.shape[0]):
            temp = np.dot(-(y-h),X[:,i])
            theta[i] = theta[i] - alpha*(1/m)*temp

    return theta
