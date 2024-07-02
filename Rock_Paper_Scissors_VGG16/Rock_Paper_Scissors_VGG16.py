import os
import numpy as np
from keras import applications, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Part 1
# a) visualized samples from the dataset, i.e.: rock, paper, scissors hand signs
# with the appropriate labels
# b) summary of the model architecture in a form of a plot or text
# c) model accuracy evaluation plot after the training concludes
# d) model loss evaluation plot after the training concludes


# Image directory's and defining the dimensions & Batch size as well as epochs
base_dir = '../rps'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
BATCH_SIZE = 32
EPOCHS = 7
img_width, img_height = 224, 224
# Define L2 regularization coefficient to prevent overfitting
l2_reg = 0.00001

# Optimization + Learning rate variables
opt = Adam(learning_rate=1e-4)
opt1 = Adam(learning_rate=2e-4)
opt2 = Adam(learning_rate=0.0001)
opt3 = SGD(learning_rate=1e-4, momentum=0.99)

# Preparing the Train/Validation and Augmentation Data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=90,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    # horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.2, 1),
    fill_mode='nearest',
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    shuffle=True,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

# a) Visualize samples from the dataset
class_names = ['paper', 'rock', 'scissors']
images, labels = train_generator.next()
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    label_index = np.argmax(labels[i])
    plt.title('Label: ' + class_names[label_index])
    plt.imshow(images[i])
    plt.tight_layout()
    plt.axis('off')
plt.show()

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# -------Callbacks-------------#
# It'll save the best trained weight
checkpoint = ModelCheckpoint(
    filepath='best_weights.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False
)
# Early stop = in case of high Validation Loss
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode='auto'
)
# Defining a learning rate reduction callback when its necessary it'll reduce
#  the learning rate when its necessary
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    cooldown=1,
    min_lr=0.000001
)
callbacks = [checkpoint, early_stop, lr_reduction]

# Load the pre-trained VGG16 model without the top layer
base_model = applications.VGG16(weights='imagenet', include_top=False, pooling='max',
                                input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers from 0-14,
# so they are not updated during training
for layer in base_model.layers[:10]:
    layer.trainable = False
# b) summary of base model
base_model.summary()

# Adding custom layers on top of VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax', kernel_regularizer=l2(l2_reg)))
# b) summary of model
model.summary()

# Compile the model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Finally we train the model with our desired adjustments
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=validation_generator)


# Plotting the Models 'accuracy' & 'loss'
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


# Evaluate the Process to find out how well the model has been trained
def evaluate(model):
    num_of_test_samples = len(validation_generator.filenames)

    y_pred = model.predict(validation_generator, num_of_test_samples // BATCH_SIZE + 1)
    y_pred = np.argmax(y_pred, axis=1)
    print('\nConfusion Matrix\n')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('\n\nClassification Report\n')
    target_names = ['Paper', 'Rock', 'Scissors']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


eval_plot(history)
evaluate(model)
model.save('../Rock_Paper_Scissors_VGG16/RPS_Model.hdf5')
