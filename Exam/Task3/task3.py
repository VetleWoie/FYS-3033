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
from sklearn.neighbors import KNeighborsClassifier
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
    convlayers = [64,"pool",128,"pool",256,256,"pool",512,512,"pool",512,512,"pool"]
    dense_layers = [4096,4096]

    model.add(layers.InputLayer(input_shape=input_shape))
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
        if l2:
            model.add(layers.Dense(nodes,activation="relu",name=f"fully_connected_{i}", kernel_regularizer=regularizers.L2()))
        else:
            model.add(layers.Dense(nodes,activation="relu",name=f"fully_connected_{i}"))
    model.add(layers.Dense(24, activation="softmax", name="output"))

    # if l2:
    #     for layer in model.layers[:-1]:
    #         if hasattr(layer, "kernel_regularizer"):
    #             layer.kernel_regularizer = regularizers.L2()
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
    label = y_true
    guess = y_pred
    perm = Permutation(4)
    label = np.argmax(label,axis=1)
    guess = np.argmax(guess,axis=1)
    sum = 0
    for i in range(label.shape[0]):
        l = np.array(perm[label[i]])
        g = np.array(perm[guess[i]])
        sum += np.sum(l == g)

    return sum / (len(label)*4)

if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_data(f"{DATAFOLDER}/{DATASHUFFLED}")

    x_tr_p=create_patched_dataset(x_tr)
    x_te_p=create_patched_dataset(x_te)
    
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

    dt,m_arg, title = model_list[int(sys.argv[1])]

    if dt:
        m = create_time_distributed_model(batch_norm=m_arg["batch_norm"], dropout=m_arg["dropout"], l2=m_arg["l2"])
    else:
        m = create_model(batch_norm=m_arg["batch_norm"], dropout=m_arg["dropout"], l2=m_arg["l2"])
    m.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["acc",axis_acc],
                    run_eagerly=True)


    batch_size = 100

    history =  m.fit(x=x_tr if not dt else x_tr_p,
                        y=y_tr,
                        validation_data=(x_te if not dt else x_te_p, y_te), 
                        batch_size=batch_size, epochs=50)
    m.save("model")
    fig, ax = plt.subplots(1,3)
    fig.suptitle(title)
    ax[0].set_title("Loss")
    ax[0].plot(history.history["loss"],label="Training")
    ax[0].plot(history.history["val_loss"],label="Validation")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("Image accuracy")
    ax[1].plot(history.history["acc"],label="Training")
    ax[1].plot(history.history["val_acc"],label="Validation")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Image Accuracy(%)")
    ax[1].plot(np.full(len(history.history["acc"]),0.6))
    ax[2].set_title("Quadrant accuracy")
    ax[2].plot(history.history["axis_acc"],label="Training")
    ax[2].plot(history.history["val_axis_acc"],label="Validation")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Quadrant Accuracy(%)")
    fig.tight_layout()
    plt.legend()
    filename = title.replace(" ", "")
    plt.savefig(f"experiments/{filename}.png")
    plt.close(fig)

    m = keras.models.load_model("model",custom_objects={"axis_acc": axis_acc})

    #3c:
    new_model = keras.Sequential()
    for layer in m.layers[:-5]:
        new_model.add(layer)
    
    x_tr, y_tr, x_te, y_te = ld(f"{DATAFOLDER}/{DATANORMAL}")
    x_tr = np.moveaxis(x_tr,1,3).astype(np.float32)/255
    x_te = np.moveaxis(x_te,1,3).astype(np.float32)/255 

    #Nearest neighbour on input images
    x_i_tr = x_tr.reshape(x_tr.shape[0],-1)
    x_i_te = x_te.reshape(x_te.shape[0],-1)
    neighbor_classifier = KNeighborsClassifier(3)
    neighbor_classifier.fit(x_i_tr, y_tr)
    print("Accuracy on input images",neighbor_classifier.score(x_i_te, y_te))

    #Nearest neighbour on learned features
    x_lf_tr = new_model.predict(x_tr)
    x_lf_te = new_model.predict(x_te)
    print(x_lf_tr.shape)
    neighbor_classifier = KNeighborsClassifier(3)
    neighbor_classifier.fit(x_lf_tr, y_tr)
    print("Accuracy on learned features",neighbor_classifier.score(x_lf_te, y_te))
    
    random_model = keras.Sequential()
    for layer in create_model(batch_norm=True, dropout=True, l2=True).layers[:-5]:
        random_model.add(layer)
    x_rf_tr = random_model.predict(x_tr)
    x_rf_te = random_model.predict(x_te)
    neighbor_classifier = KNeighborsClassifier(3)
    neighbor_classifier.fit(x_rf_tr, y_tr)
    print("Accuracy on random features",neighbor_classifier.score(x_rf_te, y_te))