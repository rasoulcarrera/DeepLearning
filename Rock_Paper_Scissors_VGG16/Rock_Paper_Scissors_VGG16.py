import os
import numpy as np
from keras import applications, Model
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import MaxPooling2D, Dense, Dropout, Conv2D, Flatten, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt, image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

# Part 1
# a) visualized samples from the dataset, i.e.: rock, paper, scissors hand signs with the appropriate
# labels
# b) summary of the model architecture in a form of a plot or text
# c) model accuracy evaluation plot after the training concludes
# d) model loss evaluation plot after the training concludes


base_dir = r'C:\MyData\DeepLearning\Rock-Paper-Scissors'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
BATCH_SIZE = 128
EPOCHS = 5

# Visualize samples from the dataset
training_scissors_dir = os.path.join(train_dir, 'rock')
sample_imgs = os.listdir(training_scissors_dir)

# Number of training, validation, and test samples
num_train_samples = 2000
num_valid_samples = 800
num_test_samples = 400
# Image dimensions & Batch size and Optimization + Learning rate
img_width, img_height = 224, 224
batch_size = 16
opt = Adam(learning_rate=0.01)
# Define a learning rate reduction callback
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

plt.figure(figsize=(20, 6))
for i, img_path in enumerate(sample_imgs[:20]):
    sp = plt.subplot(2, 10, i + 1)
    img = mpimg.imread(os.path.join(training_scissors_dir, img_path))
    plt.title(f"{sample_imgs[i]}")
    plt.axis('off')
    plt.imshow(img)
plt.show()

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

# Load the pre-trained VGG16 model without the top layer
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of VGG16
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
sub_model = Dense(3, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=sub_model)
model.summary()

# Freeze the pre-trained layers, so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=num_valid_samples // batch_size)


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
model.save(r'C:\MyData\DeepLearning\RPC_Model.hdf5')
