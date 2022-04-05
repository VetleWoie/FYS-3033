from unittest import result
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras import preprocessing
from keras import backend
from keras.losses import categorical_crossentropy

def logtransform(c, image):
    '''
    Log transforms an image
    image: Array like object
    c: float
    '''
    return c * np.log(1 + image)

def maptoKbit(image, k):
    '''
    Maps image to k bits
    image: Array like object
    k: Number of bits to be used
    '''
    return (image/np.max(image)) * (2**k-1)

if __name__ == "__main__":
    
    vgg16 = VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation=None)

    vgg16.compile()

    test_label = np.zeros((1,1000))
    test_label[0,0] = 1

    reg = 0.01

    img = np.zeros((1,224,224,3), dtype=np.float32)
    for i in range(10):
        tensor = tf.convert_to_tensor(img)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            out = vgg16(tensor, training=True)
            
            loss = backend.mean(vgg16(tensor, training=True)[:,1])-reg*backend.mean(backend.square(tensor))
            print(loss)
        grads = tape.gradient(loss, tensor)
        img += grads[0]
    
    #Unprocess image
    img = np.array(img)
    img += 1
    img *= 127.5
    img = img.astype(dtype=int)
    plt.imshow(img[0])
    plt.savefig("Goldfishmaybe.png")
    plt.show()


    # img = np.array(preprocessing.image.load_img(f"./Goldfishmaybe.png"))#,target_size=(224,224)))
    # img = preprocessing.image.smart_resize(img, (224,224))
    # img = preprocess_input(img)
    # img = np.expand_dims(img, axis=0)

    # pred = vgg16.predict(img)
    # print(decode_predictions(pred))
    # exit()