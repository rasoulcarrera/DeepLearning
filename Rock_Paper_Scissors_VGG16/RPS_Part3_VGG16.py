import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing import image

# Part 3
# a) Read two images of hand signs provided as script arguments
# b) Predict the labels of the two images
# c) Output which image won the rock, paper, scissor game

# Loading the pre-trained model/best saved weight and perform Prediction
# vgg16_model = tf.keras.models.load_model('../Rock_Paper_Scissors_VGG16/RPS_Model.hdf5')
vgg16_model = tf.keras.models.load_model('../Rock_Paper_Scissors_VGG16/best_weights.hdf5')
img_width, img_height = 224, 224
# Define labels and Image addresses
class_labels = ['paper', 'rock', 'scissors']
folder_path = '../rps/test'


# Get a random Image from Folder
def selectRandomPicture(folder_path):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random_photo = random.choice(image_files)
    return os.path.join(folder_path, random_photo)


# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Normalize pixel values

    return img_array


# A simple condition/rules to decide who’s the winner
def whos_winner(first_image, second_image):
    winner = ''
    if first_image == second_image:
        winner = "Tie!!"

    elif (first_image == 'rock' and second_image == 'scissors' or
          first_image == 'scissors' and second_image == 'rock'):
        winner = "Rock wins"

    elif (first_image == 'rock' and second_image == 'paper' or
          first_image == 'paper' and second_image == 'rock'):
        winner = "Paper wins"

    elif (first_image == 'paper' and second_image == 'scissors' or
          first_image == 'scissors' and second_image == 'paper'):
        winner = "Scissors wins"

    return winner


# Read and preprocess the images and put 2 images in separate variables
image1 = load_and_preprocess_image(selectRandomPicture(folder_path))
image2 = load_and_preprocess_image(selectRandomPicture(folder_path))

# Predict the labels of the images
images = np.array([image1, image2])
predictions = vgg16_model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)

firs_img = class_labels[predicted_classes[0]]
sec_img = class_labels[predicted_classes[1]]

# Plot the images
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title(class_labels[predicted_classes[0]])
plt.axis('off')
# Plot/Display the last result
plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.title(class_labels[predicted_classes[1]])
plt.axis('off')
plt.tight_layout()
plt.suptitle(f'{whos_winner(firs_img, sec_img)}!')
plt.show()

print(f'The winner is:{whos_winner(firs_img, sec_img)}')
