import sys

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from matplotlib import pyplot as plt

# Part 2
# a) The tested image is to be supplied via the arguments list
# b) visualisation of the supplied image with the prediction score and predicted label

# Loading the pre-trained model and perform Prediction
model = tf.keras.models.load_model(r'C:\MyData\Tensorflow\RPC_Model.hdf5')


def predict_image(image_input, model):
    if image_input is None or image_input == '' or image_input != '*.png':
        print("Invalid type")
        return None
    img = image.load_img(image_input, target_size=(100, 100))
    img_array = image.img_to_array(img)
    processed_img = tf.reshape(img_array, shape=[1, 100, 100, 3])
    #
    predict_proba = np.max(model.predict(processed_img)[0])
    predict_class = np.argmax(model.predict(processed_img))

    # # Map predicted class index to label
    class_labels = ['Paper', 'Rock', 'Scissors']
    predict_label = class_labels[predict_class]
    #
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Class: {predict_label}')
    plt.show()

    # Print prediction result and probability
    print("\nImage prediction result:", predict_label)
    print("Probability:", round(predict_proba * 100, 2), "%")
    print('\n')

    # Load the image & call Prediction
    # test_dir = r'C:\MyData\Tensorflow\Rock-Paper-Scissors\test'
    # for filename in os.listdir(test_dir):
    #     filepath = os.path.join(test_dir, filename)
    #     img = image.load_img(filepath, target_size=(100, 100))
    #
    #     predict_image(img, model)


if __name__ == "__main__":
    image_path = ''
    if len(sys.argv) != 2:
        image_path = input("Enter the path to the image file: ")
    else:
        image_path = sys.argv[1]
    predict_image(image_path, model)
