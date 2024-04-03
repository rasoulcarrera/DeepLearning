import os
import tensorflow as tf
import numpy as np
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import MaxPooling2D, Dense, Dropout, Conv2D, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from keras.preprocessing import image

base_dir = r'C:\MyData\Tensorflow\Rock-Paper-Scissors'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
BATCH_SIZE = 32
EPOCHS = 20

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.4)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                        validation_split=0.4)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 150),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(100, 150),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# Prepare the Model
model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(100, 150, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),
    Flatten(),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
#                                             patience=2,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.000003)

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


# Evaluate the Process
def evaluate(model):
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(100, 150),
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


# Prediction Function
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
# directory = r'C:\MyData\Tensorflow\Rock-Paper-Scissors\test\paper1.png'
test_dir = r'C:\MyData\Tensorflow\Rock-Paper-Scissors\test'
for filename in os.listdir(test_dir):
    filepath = os.path.join(test_dir, filename)

    img = image.load_img(filepath, target_size=(100, 150))
    # x = image.img_to_array(directory)
    # x = np.expand_dims(x, axis=0)

    predict_image(img, model)
