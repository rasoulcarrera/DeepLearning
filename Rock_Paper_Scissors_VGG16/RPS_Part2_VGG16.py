import sys
import os
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from matplotlib import pyplot as plt

# Part 2
# a) The tested image is to be supplied via the arguments list
# b) visualisation of the supplied image with the prediction score and predicted label

# Loading the pre-trained model/best saved weight and perform Prediction
# model = tf.keras.models.load_model('../Rock_Paper_Scissors_VGG16/RPS_Model.hdf5')
model = tf.keras.models.load_model('../Rock_Paper_Scissors_VGG16/best_weights.hdf5')
img_width, img_height = 224, 224


# Predict function
def predict_image(image_input, model):
    if image_input is None or image_input == '':
        print("Invalid type")
        return None
    # putting the images in an array
    img_array = image.img_to_array(image_input)
    processed_img = tf.reshape(img_array, shape=[1, img_width, img_height, 3])

    # It uses the model to predict the class probabilities for the processed image.
    predict_proba = np.max(model.predict(processed_img)[0])
    # It identifies the predicted class index and its corresponding label.
    predict_class = np.argmax(model.predict(processed_img))

    # Map predicted class index to label
    class_labels = ['Paper', 'Rock', 'Scissors']
    predict_label = class_labels[predict_class]

    # It plots the input image with its predicted class label and displays the image without axis ticks.
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Class: {predict_label}')
    plt.show()

    # Print prediction result and probability
    print("\nImage prediction result:", predict_label)
    print("Probability:", round(predict_proba * 100, 2), "%")
    print('\n')


# asking the user for their desired folder location
if __name__ == "__main__":
    image_path = ''
    if len(sys.argv) != 2:
        image_path = input("Enter the path to the image file: ")
        if input() == '':
            image_path = '../rps/test'
    # it collects 21 random images from the folder
    for filename in os.listdir(image_path)[0:20]:
        filepath = os.path.join(image_path, filename)
        # it sends the images and loaded model to prediction function
        img = image.load_img(filepath, target_size=(img_width, img_height))
        predict_image(img, model)
