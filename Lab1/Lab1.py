pip install livelossplot

import tensorflow as tf
import skimage

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import skimage.io as io
import skimage.transform

from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.models import Sequential
from livelossplot import PlotLossesKeras
import os

#МЕТОД ДЛЯ ОБРАБОТКИ ИЗОБРАЖЕНИЙ
def GetImagePath(dir):
  image_path = os.listdir(dir)
  i=0
  while i <len(image_path):
    image_path[i] = dir + image_path[i]
    i += 1
  return image_path

#ПРИВЕДЕНИЕ ТРЕНИРОВОЧНЫХ ИЗОБРАЖЕНИЙ К ЕДИНОЙ РАЗМЕРНОСТИ
uninfected_path = 'Uninfect/'
parasitized_path = 'Parasit/'

path_image = GetImagePath(uninfected_path) + GetImagePath(parasitized_path)

train_size = 400
width = 100
heigth = 100

X = np.array([])
Y = np.array([0] * train_size + [1] * train_size)
for p in path_image:
  image = io.imread(p)
  new_size = [width, heigth]
  img = skimage.transform.resize(image, new_size)
  X = np.append(X, img)

X = np.reshape(X, (len(Y), width * heigth * 3))

#ПРИВЕДЕНИЕ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ К ЕДИНОЙ РАЗМЕРНОСТИ
uninfectedTest_path = 'TestUninfect/'
parasitizedTest_path = 'TestParasit/'

path_imageTest = GetImagePath(uninfectedTest_path) + GetImagePath(parasitizedTest_path)

trainTest_size = 80
width = 100
heigth = 100

XT = np.array([])
YT = np.array([0] * trainTest_size + [1] * trainTest_size)
for p in path_imageTest:
  image = io.imread(p)
  new_size = [width, heigth]
  img = skimage.transform.resize(image, new_size)
  XT = np.append(XT, img)
    
XT = np.reshape(XT, (len(YT), width * heigth * 3))

#ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ
model = Sequential()
model.add(Dense(1000, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(250, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='SGD', metrics=['accuracy'])
history = model.fit(X, Y, batch_size=400, epochs=500, validation_split=0.1, callbacks=[PlotLossesKeras()])
model.evaluate(XT, YT, batch_size=80, callbacks=[PlotLossesKeras()])