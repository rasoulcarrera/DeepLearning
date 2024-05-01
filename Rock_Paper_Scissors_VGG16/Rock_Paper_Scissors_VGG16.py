import os
import numpy as np
from keras import applications, Model, Sequential
# from keras.src.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import MaxPooling2D, Dense, Dropout, Conv2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt, image as mpimg
from sklearn.metrics import classification_report, confusion_matrix
# from keras import callbacks

# Part 1
# a) visualized samples from the dataset, i.e.: rock, paper, scissors hand signs with the appropriate
# labels
# b) summary of the model architecture in a form of a plot or text
# c) model accuracy evaluation plot after the training concludes
# d) model loss evaluation plot after the training concludes


base_dir = '../rpc'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
BATCH_SIZE = 32
EPOCHS = 4
# Number of training, validation, and test samples
num_train_samples = 2000
num_valid_samples = 800
num_test_samples = 400

# Visualize samples from the dataset
training_scissors_dir = os.path.join(train_dir, 'rock')
sample_imgs = os.listdir(training_scissors_dir)

# Image dimensions & Batch size and Optimization + Learning rate
img_width, img_height = 224, 224
opt = Adam(learning_rate=0.01)
opt1 = SGD(learning_rate=1e-4, momentum=0.99)
opt2 = Adam(learning_rate=2e-4)
# Define a learning rate reduction callback
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

plt.figure(figsize=(6, 2))
for i, img_path in enumerate(sample_imgs[:5]):
    sp = plt.subplot(1, 5, i + 1)
    img = mpimg.imread(os.path.join(training_scissors_dir, img_path))
    plt.title(f"{sample_imgs[i].title()}")
    plt.axis('off')
    plt.imshow(img)
plt.show()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    # vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

validation_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                        validation_split=0.3)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# -------Callbacks-------------#
# best_model_weights = './base.model'
# checkpoint =  ModelCheckpoint(
#     best_model_weights,
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True,
#     mode='min',
#     save_weights_only=False
#     #period=1
# )
# earlystop = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.001,
#     patience=10,
#     verbose=1,
#     mode='auto'
# )
# tensorboard = TensorBoard(
#     log_dir='./logs',
#     histogram_freq=0,
#     batch_size=16,
#     write_graph=True,
#     write_grads=True,
#     write_images=False,
# )
#
# # csvlogger = CSVLogger(
# #     filename="training_csv.log",
# #     separator=",",
# #     append=False
# # )
# reduce = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=40,
#     verbose=1,
#     mode='auto',
#     cooldown=1
# )

# callbacks = [checkpoint, tensorboard, reduce]
# Load the pre-trained VGG16 model without the top layer
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))


# Freeze the pre-trained layers, so they are not updated during training
for layer in base_model.layers[:3]:
    layer.trainable = False

# Add custom layers on top of VGG16
# MaxPooling2D()
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
sub_model = Dense(3, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=sub_model)
model.summary()

# Compile the model
model.compile(optimizer=opt1,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator)
    # steps_per_epoch=num_train_samples // BATCH_SIZE,
    # validation_steps=num_valid_samples // BATCH_SIZE)


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
model.save('../RPC_Model.hdf5')
