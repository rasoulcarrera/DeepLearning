import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import applications, Model
from keras.src.applications import vgg16
from keras.src.layers import BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.python.keras import optimizers
from tqdm import tqdm

# Paths
train_path = r'C:\MyData\Model\catvsdog\train'
valid_path = r'C:\MyData\Model\catvsdog\train'
test_path = r'C:\MyData\Model\catvsdog\test\1'

# Number of training, validation, and test samples
num_train_samples = 2000
num_valid_samples = 800
num_test_samples = 400
opt = Adam(learning_rate=0.01)

# Image dimensions
img_width, img_height = 224, 224
batch_size = 16

# Load the pre-trained VGG16 model without the top layer
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers, so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the (lr=1e-4)  (lr=0.001)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Data augmentation for training
train_batches = (ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input).
                 flow_from_directory(train_path, target_size=(224, 224), batch_size=30))
valid_batches = (ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input).
                 flow_from_directory(valid_path, target_size=(224, 224), batch_size=30))
test_batches = (ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input).
                flow_from_directory(test_path, target_size=(224, 224), batch_size=30))
# train_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input).flow_from_directory(train_path,
#                                                                                                       target_size=(
#                                                                                                       224, 224),
#                                                                                                       batch_size=30)
# valid_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input).flow_from_directory(valid_path,
#                                                                                                       target_size=(
#                                                                                                       224, 224),
#                                                                                                       batch_size=30)
# test_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input).flow_from_directory(test_path,
#                                                                                                      target_size=(
#                                                                                                      224, 224),
#
#                                                                                                      batch_size=30)
# train_batches.classes = to_categorical(train_batches.classes)
# valid_batches.classes = to_categorical(valid_batches.classes)
# test_batches.classes = to_categorical(test_batches.classes)

test_images, test_labels = next(test_batches)

# Evaluate the model
test_results = model.evaluate(test_images, test_labels, verbose=0)
print("Test Loss:", test_results[0])
print("Test Accuracy:", test_results[1])

# test_loss, test_acc = model.evaluate(test_batches, steps=num_test_samples)
# print('Test accuracy:', test_acc)