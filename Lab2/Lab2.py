#ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt
%matplotlib inline


from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import GaussianNoise


import numpy as np
import skimage.io as io
import skimage.transform

from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
import os

#МОДЕЛЬ ДЛЯ СЧИТЫВАНИЯ ИЗОБРАЖЕНИЙ
def GetImagePath(dir):
  image_path = os.listdir(dir)
  i=0
  while i < len(image_path):
    image_path[i] = dir + image_path[i]
    i+=1
  return image_path

#ПРИВИДЕНИЕ ТРЕНИРОВОЧНОГО СЕТА К ЕДИНОЙ РАЗМЕРНОСТИ
dataSet = 'dataSet/'

path_image = GetImagePath(dataSet)

train_size = 450
width = 100
heigth = 100

X = np.array([])
Y = np.array([0] * train_size)

for p in path_image:
  image = io.imread(p)
  new_size = [width, heigth]
  img = skimage.transform.resize(image, new_size)
  X = np.append(X, img) 
  
X = np.reshape(X, (len(Y), width * heigth * 3))

#ПРИВИДЕНИЕ ТЕСТОВОГО СЕТА К ЕДИНОЙ РАЗМЕРНОСТИ И ЗАШУМЛЕНИЕ ИЗОБРАЖЕНИЙ ГАУССОВСКИМ ШУМОМ
dataSetTest = 'dataSetTest/'

path_image = GetImagePath(dataSetTest) 

train_sizeT = 50

XT = np.array([])
YT = np.array([0] * train_sizeT)

for p in path_image:
  image = io.imread(p)
  new_size = [width, heigth]
  img = skimage.transform.resize(image, new_size)
  img / 255
  sample = GaussianNoise(0.05, dtype=tf.float64)
  noisey = sample(img.astype(np.float32),training=True)
  XT = np.append(XT, noisey)
  
XT = np.reshape(XT, (len(YT), width * heigth * 3))

#МОДЕЛЬ АВТОЭНКОДЕРА
dataSetTest = 'dataSetTest/'

path_image = GetImagePath(dataSetTest) 

train_sizeT = 50

XT = np.array([])
YT = np.array([0] * train_sizeT)

for p in path_image:
  image = io.imread(p)
  new_size = [width, heigth]
  img = skimage.transform.resize(image, new_size)
  img / 255
  sample = GaussianNoise(0.05, dtype=tf.float64)
  noisey = sample(img.astype(np.float32),training=True)
  XT = np.append(XT, noisey)
  
XT = np.reshape(XT, (len(YT), width * heigth * 3))

#ОБУЧЕНИЕ АВТОЭНКОДЕРА
autoencoder.fit(X, X, epochs=30, batch_size=50, shuffle=True, validation_data=(XT, XT), callbacks=[PlotLossesKeras()])

#ФУНКЦИЯ ДЛЯ ВЫВОДА ИЗОБРАЖЕНИЙ 
def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    
    plt.figure(figsize=(2*n, 2*len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i*n + j + 1)
            plt.imshow(args[i][j].reshape(width, heigth, 3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

#ВЫВОД ЗАШУМЛЕННЫХ И РАСШУМЛЕННЫХ ИЗОБРАЖЕНИЙ
n = 10

imgs = XT[:n]

decoded_imgs = autoencoder.predict(imgs[:n], batch_size=n)
plot_digits(imgs, decoded_imgs)