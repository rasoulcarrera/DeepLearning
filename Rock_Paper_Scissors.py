import os
import tensorflow as tf
import numpy as np
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from keras.preprocessing import image

base_dir = r'C:\MyData\Tensorflow\Rock-Paper-Scissors'
train_dir = os.path.join(base_dir, 'train')

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
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Prepare the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.000003)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[learning_rate_reduction])


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
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        subset='validation')

    batch_size = 32
    num_of_test_samples = len(validation_generator.filenames)

    y_pred = model.predict(validation_generator, num_of_test_samples // batch_size + 1)
    y_pred = np.argmax(y_pred, axis=1)

    print('\nConfusion Matrix\n')
    print(confusion_matrix(validation_generator.classes, y_pred))

    print('\n\nClassification Report\n')
    target_names = ['Rock', 'Paper', 'Scissors']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


eval_plot(history)
evaluate(model)


# Prediction Function

def predict_image(image_upload):
    im = image_upload
    im_array = np.asarray(im)
    im_array = im_array * (1 / 225)
    im_input = tf.reshape(im_array, shape=[1, 100, 150, 3])

    predict_proba = sorted(model.predict(im_input)[0])[2]
    predict_class = np.argmax(model.predict(im_input))

    if predict_class == 0:
        predict_label = 'Paper'
    elif predict_class == 1:
        predict_label = 'Rock'
    else:
        predict_label = 'Scissor'

    print('\n')
    plt.figure(figsize=(4, 4))
    # plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Image predicted: {predict_label}')
    plt.show()
    print("\nImage prediction result: ", predict_label)
    print("Probability: ", round(predict_proba * 100, 2), "%")
    print('\n')


directory = r'C:\MyData\Tensorflow\Rock-Paper-Scissors\test\paper\testpaper01-00.png'
#
# for filename in os.listdir(directory):
#     filepath = os.path.join(directory, filename)
# Load the image
img = image.load_img(directory, target_size=(100, 150))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
predict_image(x)
