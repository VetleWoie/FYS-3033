from problem3.load_data import load_data as ld
from problem3.deshuffler import deshuffler
from permutation import Permutation

import tensorflow as tf
import keras
from keras import layers
from keras import losses
from keras import utils
from keras import backend
from keras import regularizers
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_extraction import image
from numba import cuda
import sys


DATAFOLDER = "problem3"
DATANORMAL = "DataNormal.npz"
DATASHUFFLED = "DataShuffled.npz"

def create_model(input_shape=(96,96,3),dropout=False,batch_norm = True, l2=False):
    """
    Creates a model after the vgg11 network.

    """ 

    model = keras.Sequential()
    convlayers = [64,"pool",128,"pool",256,"batchnorm",256,"pool",512,"batchnorm",512,"pool",512,512,"pool"]
    if batch_norm:
        convlayers.append("batchnorm")
    dense_layers = [4096,4096]

    model.add(layers.Input(shape=input_shape))
    #Convolutional layers
    for i,filters in enumerate(convlayers):
        if filters == "pool":
            model.add(layers.MaxPool2D(padding="same", name=f"Pooling_layer_{i}"))
        elif filters == "batchnorm":
            model.add(layers.BatchNormalization())
        else:
            model.add(layers.Conv2D(filters=filters,
                                    kernel_size=(3,3), 
                                    padding="same", 
                                    data_format="channels_last",
                                    activation="relu",
                                    # input_shape = None if i > 0 else input_shape,
                                    name=f"Conv_layer_{i}"))
    model.add(layers.Flatten())
    for i,nodes in enumerate(dense_layers):
        if dropout:
            model.add(layers.Dropout(0.5,name=f"dropout_{i}"))
        model.add(layers.Dense(nodes,activation="relu",name=f"fully_connected_{i}"))
    model.add(layers.Dense(24, activation="softmax", name="output"))

    if l2:
        for layer in model.layers:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = regularizers.L2()
    return model

def create_time_distributed_model(input_shape=(4,48,48,3),dropout=False,batch_norm = True, l2=False):
    """
    Creates a model after the vgg11 network.

    """ 
    model = keras.Sequential()
    convlayers = [64,"pool",128,"pool",256,256,"pool",512,512,"pool",512,512,"pool"]
    dense_layers = [4096,4096]

    model.add(layers.TimeDistributed(layers.InputLayer(input_shape=input_shape)))
    #Convolutional layers
    for i,filters in enumerate(convlayers):
        if filters == "pool":
            model.add(layers.TimeDistributed(layers.MaxPool2D(padding="same", name=f"Pooling_layer_{i}")))
        else:
            model.add(layers.TimeDistributed(layers.Conv2D(filters=filters,
                                    kernel_size=(3,3), 
                                    padding="same", 
                                    data_format="channels_last",
                                    activation="relu",
                                    # input_shape = None if i > 0 else input_shape,
                                    name=f"Conv_layer_{i}")))
            if batch_norm:
                model.add(layers.TimeDistributed(layers.BatchNormalization()))

    model.add(layers.Flatten())
    for i,nodes in enumerate(dense_layers):
        if dropout:
            model.add(layers.Dropout(0.5,name=f"dropout_{i}"))
        model.add(layers.Dense(nodes,activation="relu",name=f"fully_connected_{i}"))
    model.add(layers.Dense(24, activation="softmax", name="output"))

    if l2:
        for layer in model.layers:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = regularizers.L2()
    return model

def create_patched_dataset(dataset):
    new_ds = []
    images,height,widht,channels = dataset.shape
    for img in dataset:
        new_ds.append((
            img[:height//2,:widht//2,:],
            img[height//2:,:widht//2,:],
            img[:height//2,widht//2:,:],
            img[height//2:,widht//2:,:])
        )
    return np.array(new_ds)

def load_data(path):
    x_tr, y_tr, x_te, y_te = ld(path)
    return (np.moveaxis(x_tr,1,3).astype(np.float32)/255,
            permutation_vector_to_class_vector(y_tr), 
            np.moveaxis(x_te,1,3).astype(np.float32)/255, 
            permutation_vector_to_class_vector(y_te))

def permutation_vector_to_class_vector(vectors):
    perm = Permutation(4)
    idxs = np.array([perm.get_index(vec) for vec in vectors])
    return tf.keras.utils.to_categorical(idxs, num_classes=24)

def axis_acc(y_true, y_pred):
    perm = Permutation(4)
    print(y_true.shape)
    print(y_pred.shape)
    y_true = backend.argmax(y_true)
    y_pred = backend.argmax(y_pred)
    x = []
    for y in y_true:
        x.append(y)
    print(x)
    return np.mean(x)

if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_data(f"{DATAFOLDER}/{DATASHUFFLED}")

    x_tr_p=create_patched_dataset(x_tr)
    x_te_p=create_patched_dataset(x_te)
    
    # model = create_model(batch_norm=False, dropout=False, l2=False)
    # model_bn = create_model(batch_norm=True, dropout=False, l2=False)
    # model_do = create_model(batch_norm=False, dropout=True, l2=False)
    # model_bnl2 = create_model(batch_norm=True, dropout=False, l2=True)
    # model_bndol2 = create_model(batch_norm=True, dropout=True, l2=True)

    # dt_model = create_time_distributed_model(batch_norm=False, dropout=False, l2=False)
    # dt_model_bn = create_time_distributed_model(batch_norm=True, dropout=False, l2=False)
    # dt_model_do = create_time_distributed_model(batch_norm=False, dropout=True, l2=False)
    # dt_model_bnl2 = create_time_distributed_model(batch_norm=True, dropout=False, l2=True)
    # dt_model_bndol2 = create_time_distributed_model(batch_norm=True, dropout=True, l2=True)

    model_list = [(0,{"batch_norm":False, "dropout":False,"l2":False},"VGG 11"),
                 (0,{"batch_norm":True, "dropout":False,"l2":False},"VGG 11 with batchnorm"),
                 (0,{"batch_norm":False, "dropout":True,"l2":False},"VGG 11 with dropout"),
                 (0,{"batch_norm":True, "dropout":False,"l2":True},"VGG 11 with batchnorm and l2"),
                 (0,{"batch_norm":True, "dropout":True,"l2":True},"VGG 11 with batchnorm, dropout and l2"),
                 (1,{"batch_norm":False, "dropout":False,"l2":False},"Time distributed VGG 11"),
                 (1,{"batch_norm":True, "dropout":False,"l2":False},"Time distributed VGG 11 with batchnorm"),
                 (1,{"batch_norm":False, "dropout":True,"l2":False},"Time distributed VGG 11 with dropout"),
                 (1,{"batch_norm":True, "dropout":False,"l2":True},"Time distributed VGG 11 with batchnorm and l2"),
                 (1,{"batch_norm":True, "dropout":True,"l2":True},"Time distributed VGG 11 with batchnorm, dropout and l2"),
                ]

    for dt,m_arg, title in model_list:

        if dt:
            m = create_time_distributed_model(batch_norm=m_arg["batch_norm"], dropout=m_arg["dropout"], l2=m_arg["l2"])
        else:
            m = create_model(batch_norm=m_arg["batch_norm"], dropout=m_arg["dropout"], l2=m_arg["l2"])
        m.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=["acc"],
                        run_eagerly=True)


        batch_size = 100

        history =  m.fit(x=x_tr if not dt else x_tr_p,
                            y=y_tr,
                            validation_data=(x_te if not dt else x_te_p, y_te), 
                            batch_size=batch_size, epochs=100)
        fig, ax = plt.subplots(1,2)
        fig.suptitle(title)
        ax[0].set_title("Loss")
        ax[0].plot(history.history["loss"],label="Trainging")
        ax[0].plot(history.history["val_loss"],label="Validation")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[1].set_title("Image accuracy")
        ax[1].plot(history.history["acc"],label="Trainging")
        ax[1].plot(history.history["val_acc"],label="Validation")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Image Accuracy")
        ax[1].plot(np.full(len(history.history["acc"]),0.6))
        fig.tight_layout()
        plt.legend()
        filename = title.replace(" ", "")
        plt.savefig(f"experiments/{filename}.png")
        plt.close(fig)
        backend.clear_session()
        cuda.select_device(0)
        cuda.close()

    