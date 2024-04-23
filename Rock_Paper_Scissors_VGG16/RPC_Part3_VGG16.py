import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# Part 3
# a) Read two images of hand signs provided as script arguments
# b) Predict the labels of the two images
# c) Output which image won the rock, paper, scissor game

# Loading the pre-trained model
img_width, img_height = 224, 224
model = tf.keras.models.load_model(r'C:\MyData\DeepLearning\RPC_Model.hdf5')
# Define labels and Image addresses
class_labels = ['rock', 'paper', 'scissors']
# image1_path = r'C:\MyData\DeepLearning\Rock-Paper-Scissors\test\paper1.png'
# image2_path = r'C:\MyData\DeepLearning\Rock-Paper-Scissors\test\rock9.png'
folder_path = r'C:\MyData\DeepLearning\Rock-Paper-Scissors\test'


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


# Read and preprocess the images
image1 = load_and_preprocess_image(selectRandomPicture(folder_path))
image2 = load_and_preprocess_image(selectRandomPicture(folder_path))

# Plot the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.title('Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()

# Predict the labels of the images
images = np.array([image1, image2])
predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted Image 1:", class_labels[predicted_classes[0]])
print("Predicted Image 2:", class_labels[predicted_classes[1]])

# Determine the winner of the rock, paper, scissors game
winner_index = np.argmax(predictions.sum(axis=0))
winner_label = class_labels[winner_index]
print("Winner is: ", winner_label)
