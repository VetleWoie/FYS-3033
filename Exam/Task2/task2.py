from cProfile import label
from re import I
from unittest import result
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow.keras.utils as tfutils

PRECODEDIR = "problem2"
TESTIAMGES = "test_images"
VALIDATIONIMAGES = "validation_images"
VALIDATIONFILES = "val.txt"
SYNSETWORDS = "synset_words.txt"


def load_test_images():
    pass

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

def load_validation_images(reshape_img = False, expected_shape = (224,224,3)):
    x = []
    y = []
    with open(f"{PRECODEDIR}/{SYNSETWORDS}") as file:
        num_classes = len(file.readlines())
    with open(f"{PRECODEDIR}/{VALIDATIONFILES}", 'r') as file:
        filenames = file.readlines()
        
        print(f"{PRECODEDIR}/{VALIDATIONIMAGES}/{filenames[0].split(' ')[0]}")
        for filename in filenames:
            filename, c = filename.split(" ")
            img = Image.open(f"{PRECODEDIR}/{VALIDATIONIMAGES}/{filename}")
            if reshape_img:
                img = isotropic_reshape(img, expected_shape)
            img = np.array(img)/255
            if reshape_img:
                #Crop largest axis to wanted size
                if img.shape[0] > img.shape[1]:
                    img = img[:expected_shape[0], :]
                else:
                    img = img[:,:expected_shape[1]]
            
            x.append(img)
            y.append(int(c))
    return x,tfutils.to_categorical(y, num_classes)

def load_validation_images_from_dir(dir):
    labels = []
    with open(f"./{PRECODEDIR}/{VALIDATIONFILES}", 'r') as file:
        filenames = file.readlines()
        for filename in filenames:
            filename, c = filename.split(" ")
            labels.append(int(c))
    print(labels)

    x = tfutils.image_dataset_from_directory(directory=dir, 
                                        labels=labels,
                                        label_mode="categorical",
                                        color_mode="rgb",
                                        crop_to_aspect_ratio=True,
                                        image_size=(224,224))
    print(x)

def show_validation_images(images, classes, duration = 0.3):
    fig, ax = plt.subplots(1,1)
    for img, c in zip(images, classes):
        i = np.where(c == 1)[0][0]
        print(i)
        try:
            ax.imshow(img)
            ax.set_title(f"Class: {i}, shape = {img.shape}")
            plt.pause(duration)
        except Exception as e:
            print(e)
            plt.close(fig)
            break

if __name__ == "__main__":
    # images, classes = load_validation_images(reshape_img=True)
    load_validation_images_from_dir(f"./{PRECODEDIR}/test")
    exit()

    # show_validation_images(images, classes)
    print("Image shape:",images[0].shape)
    # print(images)

    vgg16 = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=len(classes[0]),
    classifier_activation='softmax')

    vgg16.compile()

    print(vgg16.summary())

    print(len(images), len(classes))
    print(classes[0].shape)
    print("Images shape",images.shape)
    res = vgg16.evaluate(images[:1], classes[:1])
    print("Loss, Acc", res)

    # for img, c in zip(images[:1], classes[:1]):
    #     img = img.reshape(1,224,224,3)
    #     print(img.shape, c.T.shape)
    #     print(vgg16.predict(img).shape)