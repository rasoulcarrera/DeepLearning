import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing import image

# Part 2
# a) The tested image is to be supplied via the arguments list
# b) visualisation of the supplied image with the prediction score and predicted label

# Loading the pre-trained model and perform Prediction
model = tf.keras.models.load_model(r'C:\MyData\Tensorflow\RPC_Model.hdf5')


def predict_image(test_img, model):
    im_array = np.asarray(test_img)
    im_array = im_array * (1 / 225)
    im_input = tf.reshape(im_array, shape=[1, 100, 150, 3])

    predict_proba = np.max(model.predict(im_input)[0])
    predict_class = np.argmax(model.predict(im_input))

    # Map predicted class index to label
    class_labels = ['Paper', 'Rock', 'Scissors']
    predict_label = class_labels[predict_class]

    plt.figure(figsize=(4, 4))
    plt.imshow(test_img)
    plt.axis('off')
    plt.title(f'Predicted Class: {predict_label}')
    plt.show()

    # Print prediction result and probability
    print("\nImage prediction result:", predict_label)
    print("Probability:", round(predict_proba * 100, 2), "%")
    print('\n')


# Load the image & call Prediction
test_dir = r'C:\MyData\Tensorflow\Rock-Paper-Scissors\test'
for filename in os.listdir(test_dir):
    filepath = os.path.join(test_dir, filename)
    img = image.load_img(filepath, target_size=(100, 150))

    predict_image(img, model)
