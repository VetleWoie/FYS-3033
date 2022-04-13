from pyexpat import model
from unittest import result
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras import preprocessing
from keras import layers
import keras
import os

PRECODEDIR = "problem2"
TESTIMAGES = "test_images"
VALIDATIONIMAGES = "validation_images"
VALIDATIONFILES = "val.txt"
SYNSETWORDS = "synset_words.txt"

def logtransform(c, image):
    '''
    Log transforms an image
    image: Array like object
    c: float
    '''
    return c * np.log(1 + image)

def load_test_images():
    filenames  = next(os.walk(f"{PRECODEDIR}/{TESTIMAGES}"))[2]
    x = []
    for filename in filenames:
        filename= filename
        img = np.array(preprocessing.image.load_img(f"{PRECODEDIR}/{TESTIMAGES}/{filename}"))#,target_size=(224,224)))
        img = preprocessing.image.smart_resize(img, (224,224))
        img = preprocess_input(img)
        x.append(img)
    return np.asarray(x), filenames

def isotropic_reshape(img, shape):
    #Reshape smallest axis to wanted size
    if img.size[0] > img.size[1]:
        relation = shape[1] / img.size[1]
        new_size = (int(relation * img.size[0]), int(relation * img.size[1]))
    else:
        relation = shape[0] / img.size[0]
        new_size = (int(relation * img.size[0]), int(relation * img.size[1]))
    img = img.resize(new_size)
    return img

def create_one_hot(val, max):
    onehot = np.zeros(max)
    onehot[val] = 1
    return onehot.reshape(1,-1)

def load_validation_images(preprocess=False):
    x = []
    y = []
    with open(f"{PRECODEDIR}/{SYNSETWORDS}") as synset_file:
        words = synset_file.readlines()
    with open(f"{PRECODEDIR}/{VALIDATIONFILES}", 'r') as file:
        filenames = file.readlines()
        print(f"{PRECODEDIR}/{VALIDATIONIMAGES}/{filenames[0].split(' ')[0]}")
        for filename in filenames:
            filename, c = filename.split(" ")
            img = np.array(preprocessing.image.load_img(f"{PRECODEDIR}/{VALIDATIONIMAGES}/{filename}"))#,target_size=(224,224)))
            img = preprocessing.image.smart_resize(img, (224,224))
            if preprocess:
                img = preprocess_input(img)
            x.append(img)
            y.append((int(c), words[int(c)].split()[0]))
    return np.asarray(x),np.array(y)
    
def load_synset_words():
    ids = []
    word_dict = {}
    with open(f"{PRECODEDIR}/{SYNSETWORDS}") as file:
        words = file.readlines()
    for word in words:
        word = word.split(" ")
        id = word[0]
        ids.append(id)
        word_dict[id] = word[1:]
    return ids, word_dict

        
def show_validation_images(images, classes, duration = 0.3):
    fig, ax = plt.subplots(1,1)
    for img, c in zip(images, classes):
        try:
            ax.imshow(img[0])
            ax.set_title(f"Class: {c}, shape = {img.shape}")
            plt.pause(duration)
        except Exception as e:
            print(e)
            plt.close(fig)
            break
    plt.close(fig)

def load_model():
    vgg16 = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax')
    print(vgg16.summary())
    weights = vgg16.get_weights()
    print(len(weights))

    model = tf.keras.Sequential()
    # Block 1
    model.add(layers.Conv2D(
        64, (3, 3),input_shape=(224,224,3), activation='relu', padding='same', name='block1_conv1'))
    model.add(layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dense(4096, activation='relu', name='fc2'))
    model.add(layers.Dense(1000, activation="softmax",
                     name='predictions'))
    
    model.set_weights(vgg16.get_weights())
    return model

def validation_accuracy(model):
    images, labels = load_validation_images(preprocess=True)
    pred = model.predict(images)
    pred = np.array(decode_predictions(pred, top=1)).reshape(95,3)
    correct = (pred[:,0] == labels[:,1])
    print(f"Accuracy: {correct.sum()}/{len(correct)} = {correct.sum()/len(correct)}")
    return correct.sum()/len(correct)

def predict_on_test_image(model):
    images, filenames = load_test_images()
    preds = model.predict(images)
    preds = np.array(decode_predictions(preds, top=5))
    for pred, filename in zip(preds, filenames):
        print(f"Predictions for {filename}:")
        for p in pred:
            print(p)
        print()
    return images, filenames, preds

def unprocess_image(img):
    mean = [103.939, 116.779, 123.68]
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    img = img[..., ::-1]
    img = img.astype(int)
    return img

@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy>0, np.float32) * tf.cast(x>0, np.float32) * dy
    return tf.nn.relu(x), grad

def compute_saliency_maps(model, show = False, savefig = True, guided = False):
    images, filenames, preds = predict_on_test_image(model)
    ids, word_dir = load_synset_words()
    ids = np.array(ids)

    if guided:
        for layer in model.layers:
            if hasattr(layer, "activation"):
                if layer.activation == keras.activations.relu:
                    layer.activation = guided_relu

    for filename,(img, pred) in zip(filenames,zip(images, preds)):
        class_number = np.where(ids == pred[0][0])[0][0]
        img = np.expand_dims(img, axis=0)
        tensor = tf.convert_to_tensor(img)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            score = model(tensor)[:,class_number]
        grads = tape.gradient(score, tensor)
        if show or savefig:
            fig, ax = plt.subplots(1,2)
            ax[0].set_title(f"{filename}")
            ax[0].imshow(unprocess_image(img[0]))
            grads = np.max(grads[0], axis=2)
            ax[1].imshow(np.array(grads))
            plt.show()


if __name__ == "__main__":
    vgg16 = load_model()
    vgg16.compile()
    print(vgg16.summary())
    # validation_accuracy(vgg16)
    predict_on_test_image(vgg16)
    compute_saliency_maps(vgg16, guided=True)
    