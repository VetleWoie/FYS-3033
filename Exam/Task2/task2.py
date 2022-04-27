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
from scipy.ndimage import gaussian_filter
from keras.regularizers import L2



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

def load_model(softmax = True):
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

    # Classification block with dropout layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000, activation="softmax" if softmax else None,
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
    
    return images, filenames, preds

def unprocess_image(img):
    #Unprocess image after imagenet preprocessing
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
        #Remove all negative gradients
        return tf.cast(dy>0, np.float32) * tf.cast(x>0, np.float32) * dy
    #Return regular forward pass of relu, but guided relus gradient function
    return tf.nn.relu(x), grad

def compute_saliency_maps(model,k=500, show = False, savefig = True, guided = True, out_dir="./Figures"):
    #Make a prediction on all test images
    images, filenames, preds = predict_on_test_image(model)
    #Load all synset words
    ids, word_dir = load_synset_words()
    ids = np.array(ids)
    #Calculate uncertainty of guesses on test images
    uncertainties = calculate_uncertainty(model, images, k=k)

    #Create copy of model with guided backpropogation
    if guided:
        guided_relu_model = load_model(softmax=True)
        for layer in guided_relu_model.layers:
            if hasattr(layer, "activation"):
                if layer.activation == keras.activations.relu:
                    layer.activation = guided_relu

    #Ugliest for loop in the world
    for uncertainty,(filename,(img, pred)) in zip(uncertainties,zip(filenames,zip(images, preds))):
        #Extract uncertaint measurments
        new_pred = []
        for i,p in enumerate(pred):
            class_number = np.where(ids == p[0])[0][0]
            new_pred.append(np.append(p, (uncertainty[0][class_number],uncertainty[1][class_number])))

        #Find top prediction of image
        pred = np.array(new_pred)
        class_number = np.where(ids == pred[0][0])[0][0]
        img = np.expand_dims(img, axis=0)
        tensor = tf.convert_to_tensor(img)

        #Compute saliancy map on image and top prediction
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            score = model(tensor)[:,class_number]
        unguided_grads = np.abs(tape.gradient(score, tensor))

        #Compute saliancy maps using guided backpropogation
        if guided:
            with tf.GradientTape() as tape:
                tape.watch(tensor)
                score = guided_relu_model(tensor)[:,class_number]
            guided_grads = np.abs(tape.gradient(score, tensor))
        #Create figure
        if show or savefig:
            fig, ax = plt.subplots(2,2)
            for axis in ax:
                for a in axis:
                    a.axis("off")
            ax[0][0].set_title(f"{filename}")
            ax[0][0].imshow(unprocess_image(img[0]))
            ax[0][1].axis('tight')
            ax[0][1].set_title("Top 5 Class scores")
            ax[0][1].table(cellText=pred[:,1:],colLabels=["Predictions", "Score", f"Avarage {k}", f"Uncertainty"], loc='center')      
            unguided_grads = np.max(unguided_grads, axis=3)[0]
            ax[1][0].set_title(f"Unguided backprop")
            ax[1][0].imshow(np.array(unguided_grads),cmap="jet")
            if guided:
                guided_grads = np.max(guided_grads, axis=3)[0]
                ax[1][1].set_title(f"Guided backprop")
                ax[1][1].imshow(np.array(guided_grads),cmap="jet")

            if savefig:
                plt.savefig(f"{out_dir}/{filename}_k{k}_saliancy_uncertainty.pdf")
            if show:
                plt.show()
            plt.close(fig)

def calculate_uncertainty(model, images, k=500):
    data = []
    for img in images:
        preds = []
        for i in range(k):
            pred = model(np.expand_dims(img, axis=0),training=True)
            preds.append(np.array(pred).reshape((-1)))
        preds = np.array(preds)
        mean = np.mean(preds,axis=0)
        var = np.var(preds,axis=0)
        data.append((mean, var))
    return data

def class_model_visualisation(model, 
                            class_number,
                            num_iterations=1000,
                            apply_gausian = True,
                            gausian_step = 5,
                            show_image=False, 
                            save_image=True, 
                            save_dir=".",
                            reg_class=L2,
                            learning_rate=1,
                            reg_param=0.01):
    # img = np.random.randint(0,255,(1,224,224,3))
    # img = preprocess_input(img)
    img = np.zeros((1,224,224,3), dtype=np.float32)
    img_gaus = np.zeros((1,224,224,3), dtype=np.float32)

    reg = reg_class(reg_param)
    for i in range(num_iterations):
        tensor = tf.convert_to_tensor(img)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            score = model(tensor)[:,class_number]-reg(tensor)
        print(i, "score: ",score)
        grads = tape.gradient(score, tensor)
        img += learning_rate*grads[0]
    
    if apply_gausian:
        for i in range(num_iterations):
            tensor = tf.convert_to_tensor(img_gaus)
            with tf.GradientTape() as tape:
                tape.watch(tensor)
                score = model(tensor)[:,class_number]-reg(tensor)
            print(i, "score: ",score)
            grads = tape.gradient(score, tensor)
            img_gaus += learning_rate*grads[0]
            if i % gausian_step == 0:
                img_gaus = gaussian_filter(img_gaus, sigma=0.5)


    guess = model.predict(img)
    guess = decode_predictions(guess, top=1)
    guess_gaus = model.predict(img_gaus)
    guess_gaus = decode_predictions(guess_gaus, top=1)

    if show_image or save_image:
        fig, ax = plt.subplots(1,2)
        new_img = np.array(img[0])
        new_img_gaus = np.array(img_gaus[0])
        new_img += 1
        new_img *= 127.5
        new_img_gaus += 1
        new_img_gaus *= 127.5
        new_img = new_img.astype(dtype=int)
        new_img_gaus = new_img_gaus.astype(dtype=int)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].set_title(f"Guess: {guess[0][0][1]} %.2f"%guess[0][0][2])
        ax[0].imshow(new_img)
        ax[1].set_title(f"Guess: {guess_gaus[0][0][1]} %.2f"%guess_gaus[0][0][2])
        ax[1].imshow(new_img_gaus)
        if save_image:
            gausian_string = "with_gausian" if apply_gausian else ""
            plt.savefig(f"{save_dir}/{class_number}_visualized.pdf")
        if show_image:
            plt.show()

if __name__ == "__main__":
    #https://towardsdatascience.com/monte-carlo-dropout-7fd52f8b6571
    vgg16 = load_model(softmax = False)
    vgg16.compile()
    print(vgg16.summary())
    class_model_visualisation(vgg16,2,learning_rate=1,num_iterations=1000,apply_gausian=True, show_image=False, save_image=True)
    class_model_visualisation(vgg16,215,learning_rate=1,num_iterations=1000,apply_gausian=True, show_image=False, save_image=True)
    class_model_visualisation(vgg16,659,learning_rate=1,num_iterations=1000,apply_gausian=True, show_image=False, save_image=True)


    # validation_accuracy(vgg16)
    #predict_on_test_image(vgg16)
    # compute_saliency_maps(vgg16,k=10,show=True, savefig=True, guided=True)
    