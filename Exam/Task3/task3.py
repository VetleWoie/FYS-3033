from problem3.load_data import load_data as ld
from problem3.deshuffler import deshuffler

import tensorflow as tf
import keras
from keras import layers
from matplotlib import pyplot as plt
import numpy as np

DATAFOLDER = "problem3"
DATANORMAL = "DataNormal.npz"
DATASHUFFLED = "DataShuffled.npz"

def create_model(input_shape=(3,96,96),dropout=False,batch_norm = True):
    """
    Creates a model after the vgg11 network.

    """ 

    model = keras.Sequential()
    convlayers = [64,"pool",128,"pool",256,256,"pool",512,512,"pool",512,512,"pool"]
    dense_layers = [4096,4096,4]

    # model.add(layers.Input(shape=input_shape))
    #Convolutional layers
    for i,filters in enumerate(convlayers):
        if filters == "pool":
            model.add(layers.MaxPool2D(padding="same", name=f"Pooling_layer_{i}"))
        else:
            model.add(layers.Conv2D(filters=filters,
                                    kernel_size=(3,3), 
                                    padding="same", 
                                    data_format="channels_last",
                                    activation="relu",
                                    # input_shape = None if i > 0 else input_shape,
                                    name=f"Conv_layer_{i}"))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    for i,nodes in enumerate(dense_layers):
        if dropout:
            model.add(layers.Dropout(0.5,name=f"dropout_{i}"))
        model.add(layers.Dense(nodes,activation="relu",name=f"fully_connected_{i}"))
    return model

def load_data(path):
    x_tr, y_tr, x_te, y_te = ld(path)
    return np.moveaxis(x_tr,1,3), y_tr, np.moveaxis(x_te,1,3), y_te

def image_acc():
    pass

if __name__ == "__main__":
    data = load_data(f"{DATAFOLDER}/{DATASHUFFLED}")
    model = create_model()
    model.compile()
    testimg = data[0][0].astype(np.float32)/255
    testimg = np.expand_dims(testimg, axis=0)
    print(testimg.shape)
    pred = model.predict(testimg)
    print(pred)
    