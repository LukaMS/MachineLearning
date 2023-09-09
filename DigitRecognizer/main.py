import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from functions import *
from sklearn.model_selection import train_test_split

train = pd.read_csv('DigitRecognizer/train.csv')
test = pd.read_csv('DigitRecognizer/test.csv')

X = train.drop('label', axis = 1)
X = X.to_numpy()
X = np.reshape(X,(-1,28,28,1))

Xtest = test.to_numpy()
Xtest = np.reshape(Xtest, (-1,28,28,1))

y = train['label'].to_numpy()
#y = np.reshape(y, (-1,1))

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.2)


sub = pd.read_csv('DigitRecognizer/submission.csv').to_numpy()
sub2 = pd.DataFrame({'ImageId': sub[:,1], 'Label': sub[:,2]})
sub2.to_csv('Submission.csv', index = False)
# xtrain, ytrain = createLargerDataset(xtrain, ytrain)


# model = Sequential(
#     [
#         tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape=(28,28,1)),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Flatten(),
#         Dense(units = 45, activation = 'relu'),
#         Dense(units = 25, activation = 'relu'),
#         Dense(units = 10, activation = 'softmax')
#     ]
# )

# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=['accuracy']
# )

# model.fit(
#     xtrain,ytrain,
#     epochs = 40,
#     validation_data=(xtest, ytest)
# )

# predictions = model.predict(Xtest)
# predictions = [np.argmax(pred) for pred in predictions]

# submission = pd.read_csv('DigitRecognizer/sample_submission.csv')
# submission['Label'] = predictions
# submission.to_csv('DigitRecognizer/submission.csv')
# makePredictions(model, Xtest)