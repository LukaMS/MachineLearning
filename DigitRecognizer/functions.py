import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def createLargerDataset(xtrain, ytrain):
  datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    height_shift_range=0.1,
    zoom_range=0.2,  
  )
  new_images = []
  new_labels = []
  num_new_per_image = 5
  for x_batch, y_batch in datagen.flow(xtrain, ytrain, batch_size=num_new_per_image):
    for x_new, y_new in zip(x_batch, y_batch):
        new_images.append(x_new)
        new_labels.append(y_new)
    if len(new_images) >= len(xtrain) * num_new_per_image:
        break
  xtrain = np.concatenate((xtrain, new_images))
  ytrain = np.concatenate((ytrain, new_labels))

  return xtrain, ytrain
def makePredictions(model, X):
    m = X.shape[0]
    predictions = model.predict(X)
    newPredictions = np.zeros(m)
    for i in range(m):
        prediction = np.argmax(predictions[i])
        newPredictions[i] = prediction
    Ids = np.arange(1,28001)
    newPredictions = newPredictions.astype('int64')
    output = pd.DataFrame({'ImageId': Ids, 'Label': newPredictions})
    output.to_csv('submission.csv', index=False)
