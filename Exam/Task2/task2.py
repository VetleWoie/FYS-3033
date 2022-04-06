from pyexpat import model
from unittest import result
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras import preprocessing
import os

PRECODEDIR = "problem2"
TESTIMAGES = "test_images"
VALIDATIONIMAGES = "validation_images"
VALIDATIONFILES = "val.txt"
SYNSETWORDS = "synset_words.txt"


def load_test_images():
    filenames  = next(os.walk(f"{PRECODEDIR}/{TESTIMAGES}"))[2]
    x = []
    for filename in filenames:
        filename= filename
        img = np.array(preprocessing.image.load_img(f"{PRECODEDIR}/{VALIDATIONIMAGES}/{filename}"))#,target_size=(224,224)))
        img = preprocessing.image.smart_resize(img, (224,224))
        x.append(img)
    return np.asarray(x)

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
    return tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax')

def validation_accuracy(model):
    images, labels = load_validation_images(preprocess=True)
    ids, word_dir = load_synset_words()
    pred = model.predict(images)
    pred = np.array(decode_predictions(pred, top=1)).reshape(95,3)
    correct = (pred[:,0] == labels[:,1])
    print(f"Accuracy: {correct.sum()}/{len(correct)} = {correct.sum()/len(correct)}")
    return correct.sum()/len(correct)

def compute_saliency_map(image, show = False, savefig = True):




if __name__ == "__main__":
    vgg16 = load_model()
    vgg16.compile()
    print(vgg16.summary())
    validation_accuracy(vgg16)

    