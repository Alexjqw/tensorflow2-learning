import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test)= fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]


print(x_valid.shape,y_valid.shape)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)



class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self,units,activation = None,**kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer,selft).__init__(**kwargs)
    deff build(selt,input_shape):
        self..kernel = self.add_weight(name = 'kernel',shape = (input_shape[1],self..units),initializer  = 'uniform',trainable = True)
        self.bias = self.add_weight(name = 'bias',shape = (sl))

        pass
    deff call(selt,x):
        pass

model = keras.models.Sequential([
   keras.layers.Flatten(input_shape = [28,28]),
   keras.layers.Dense(300,activation='relu'),
   keras.layers.Dense(100,activation='relu'),
   keras.layers.Dense(10,activation = 'softmax')
]
)
model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics = ["accuracy"])

logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
out_model_file = os.path.join(logdir,"fashion_minst_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(out_model_file,save_best_only = True),
    keras.callbacks.EarlyStopping(patience = 5,min_delta = 1e-3),
]
history = model.fit(x_train_scaled,y_train,epochs =10,validation_data = (x_valid_scaled,y_valid),callbacks = callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,3)
    plt.show()

plot_learning_curves(history)

model.evaluate(x_test_scaled,y_test)

