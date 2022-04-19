from problem3.load_data import load_data as ld
from problem3.deshuffler import deshuffler
from itertools import permutations

import tensorflow as tf
import keras
from keras import layers
from keras import losses
from keras import utils
from matplotlib import pyplot as plt
import numpy as np

DATAFOLDER = "problem3"
DATANORMAL = "DataNormal.npz"
DATASHUFFLED = "DataShuffled.npz"

def create_model(input_shape=(96,96,3),dropout=False,batch_norm = True):
    """
    Creates a model after the vgg11 network.

    """ 

    model = keras.Sequential()
    convlayers = [64,"pool",128,"pool",256,256,"pool",512,512,"pool",512,512,"pool"]
    dense_layers = [4096,4096]

    model.add(layers.Input(shape=input_shape))
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
    model.add(layers.Dense(24, activation="softmax", name="output"))
    return model

def load_data(path):
    x_tr, y_tr, x_te, y_te = ld(path)
    return (np.moveaxis(x_tr,1,3).astype(np.float32)/255,
            permutation_vector_to_class_vector(y_tr), 
            np.moveaxis(x_te,1,3).astype(np.float32), 
            permutation_vector_to_class_vector(y_te)/255)

class PermutationIterator():
    def __init__(self, permutation) -> None:
        self._permutation = permutation
        self._index = 0
    
    def __next__(self):
        print("Iter",self._index,"len:" ,len(self._permutation))
        if self._index >= len(self._permutation):
            print("stop!")
            raise StopIteration
        else:
            result = self._permutation[self._index]
            self._index += 1
            return result

class Permutation():
    def __init__(self, max) -> None:
        self.max = max
        self.perms = [perm for perm in permutations(np.arange(max))]
        self._index = 0 

    def get_index(self, perm):
        perm = tuple(perm)
        for i, p in enumerate(self.perms):
            if p == perm:
                return i

        raise ValueError("No such permutation")

    def __len__(self):
        return len(self.perms)
    
    def __getitem__(self, key):
        return self.perms[key]

    def __iter__(self):
        return PermutationIterator(self)
    
    def __str__(self) -> str:
        return str(self.perms)


def permutation_vector_to_class_vector(vectors):
    perm = Permutation(4)
    idxs = np.array([perm.get_index(vec) for vec in vectors])
    return tf.keras.utils.to_categorical(idxs, num_classes=24)

if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_data(f"{DATAFOLDER}/{DATASHUFFLED}")
    model = create_model(batch_norm=True, dropout=True)

    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["acc"])
    print(model.summary())

    history =  model.fit(x=x_tr,
                        y=y_tr,
                        validation_data=(x_te, y_te), 
                        batch_size=100, epochs=100)

    testimg = x_tr[0]/255
    testimg = np.expand_dims(testimg, axis=0)
    print(testimg.shape)
    pred = model.predict(testimg)
    print(pred)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(history.history["loss"],label="Trainging image loss")
    ax[0].plot(history.history["val_loss"],label="Validation image loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].plot(history.history["acc"],label="Trainging image accuracy")
    ax[1].plot(history.history["val_acc"],label="Validation image accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Image Accuracy")
    plt.savefig("acc.png")
    plt.close(fig)
    