from unittest import result
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras import preprocessing
from keras import backend
from keras.regularizers import L2
from keras.losses import categorical_crossentropy
from scipy.ndimage import gaussian_filter

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
    # img = np.random.rand(1,224,224,3)*2-1
    img = np.zeros((1,224,224,3), dtype=np.float32)

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
            if i % gausian_step == 0:
                img = gaussian_filter(img, sigma=0.5)


    guess = model.predict(img)
    guess = decode_predictions(guess, top=1)
    if show_image or save_image:
        fig, ax = plt.subplots(1,1)
        new_img = np.array(img[0])
        new_img += 1
        new_img *= 127.5
        new_img = new_img.astype(dtype=int)
        ax.set_title(f"Guess: {guess}")
        ax.imshow(new_img)
        if save_image:
            gausian_string = "with_gausian" if apply_gausian else ""
            plt.savefig(f"{save_dir}/{class_number}_visualized_{gausian_string}.pdf")
        if show_image:
            plt.show()

if __name__ == "__main__":
    
    vgg16 = VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation=None)

    class_model_visualisation(vgg16,2,learning_rate=5,num_iterations=3000,apply_gausian=False, show_image=True, save_image=False)
    class_model_visualisation(vgg16,2,learning_rate=5,num_iterations=3000,apply_gausian=True, show_image=True, save_image=False)

    class_model_visualisation(vgg16,215,learning_rate=5,num_iterations=3000,apply_gausian=False, show_image=True, save_image=False)
    class_model_visualisation(vgg16,215,learning_rate=5,num_iterations=3000,apply_gausian=True, show_image=True, save_image=False)

    class_model_visualisation(vgg16,659,learning_rate=5,num_iterations=3000,apply_gausian=False, show_image=True, save_image=False)
    class_model_visualisation(vgg16,659,learning_rate=5,num_iterations=3000,apply_gausian=True, show_image=True, save_image=False)
    exit()
    reg = 0.01
    l2 = L2(reg)

    img = np.zeros((1,224,224,3), dtype=np.float32)
    for i in range(1000):
        tensor = tf.convert_to_tensor(img)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            score = vgg16(tensor)[:,2]-l2(tensor)
            print(i, "score: ",score)
        grads = tape.gradient(score, tensor)
        img += grads[0]
    
    #Unprocess image
    new_img = np.array(img)
    new_img += 1
    new_img *= 127.5
    new_img = new_img.astype(dtype=int)
    plt.imshow(new_img[0])
    plt.savefig("Goldfishmaybe.png")
    plt.show()

    