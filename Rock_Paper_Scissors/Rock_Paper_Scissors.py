import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import MaxPooling2D, Dense, Dropout, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt, image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

# Part 1
# a) visualized samples from the dataset, i.e.: rock, paper, scissors hand signs with the appropriate
# labels
# b) summary of the model architecture in a form of a plot or text
# c) model accuracy evaluation plot after the training concludes
# d) model loss evaluation plot after the training concludes


base_dir = '../rps'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
BATCH_SIZE = 32
EPOCHS = 20
img_width, img_height = 224, 224

# Visualize samples from the dataset
test_dir = os.path.join(base_dir, 'test')
random_images = random.sample(os.listdir(test_dir), 10)

plt.figure(figsize=(8, 2))
for i, img_path in enumerate(random_images[:5]):
    sp = plt.subplot(1, 5, i + 1)
    img = mpimg.imread(os.path.join(test_dir, img_path))
    plt.title(f"{random_images[i].title()}")
    plt.axis('off')
    plt.imshow(img)
plt.show()

# Preparing the Train/Validation and Augmentation Data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    shear_range=0.2,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                        validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# Prepare the Model using Convolutional Neural Network (CNN) architecture
model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
#                                             patience=2,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.000003)

# We compile the model and train it with help of 'model.fit' function
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator)


# callbacks=[learning_rate_reduction])


# Plotting the Model
def eval_plot(history):
    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    acc_plot, = plt.plot(epochs, acc, 'r')
    val_acc_plot, = plt.plot(epochs, val_acc, 'b')
    plt.title('Training and Validation Accuracy')
    plt.legend([acc_plot, val_acc_plot], ['Training Accuracy', 'Validation Accuracy'])

    # Loss plot
    plt.subplot(1, 2, 2)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    loss_plot, = plt.plot(epochs, loss, 'r')
    val_loss_plot, = plt.plot(epochs, val_loss, 'b')
    plt.title('Training and Validation Loss')
    plt.legend([loss_plot, val_loss_plot], ['Training Loss', 'Validation Loss'])
    plt.tight_layout()
    plt.show()


# Evaluate the Process
def evaluate(model):
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation')

    num_of_test_samples = len(validation_generator.filenames)

    y_pred = model.predict(validation_generator, num_of_test_samples // BATCH_SIZE + 1)
    y_pred = np.argmax(y_pred, axis=1)

    print('\nConfusion Matrix\n')
    print(confusion_matrix(validation_generator.classes, y_pred))

    print('\n\nClassification Report\n')
    target_names = ['Rock', 'Paper', 'Scissors']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


eval_plot(history)
evaluate(model)
model.save('../Rock_Paper_Scissors/RPC_Model.hdf5')
