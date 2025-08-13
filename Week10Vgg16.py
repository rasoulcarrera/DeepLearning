import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import applications, Model
from keras.src.layers import BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from tensorflow.python.keras import optimizers
from tqdm import tqdm
from keras import applications, Model

# Path to the directories containing your data
train_path = r'...\train'
valid_path = r'...\train'
test_path = r'...\test'

# Number of training, validation, and test samples
num_train_samples = 2000
num_valid_samples = 800
num_test_samples = 400
opt = Adam(learning_rate=0.01)

# path to the model weights files.
# Image dimensions
img_width, img_height = 224, 224
batch_size = 16

# Load the pre-trained VGG16 model without the top layer
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Create a new model
# model = Sequential()
# # Add the VGG16 base model
# model.add(base_model)

# Add custom layers on top of VGG16
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
sub_model = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=sub_model)
model.summary()

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model = Model(input=base_model.input, output=top_model(base_model.output))

# Freeze the pre-trained layers, so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the (lr=1e-4)  (lr=0.001)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# No data augmentation for validation and test
valid_test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generate batches of augmented data for training
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
# Generate batches of data for validation and test
valid_generator = valid_test_datagen.flow_from_directory(valid_path,
                                                         target_size=(img_width, img_height),
                                                         batch_size=batch_size,
                                                         class_mode='binary')

test_generator = valid_test_datagen.flow_from_directory(test_path,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
# train_batches = (ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input).
#                  flow_from_directory(train_path, target_size=(224, 224), batch_size=30))
# valid_batches = (ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input).
#                  flow_from_directory(valid_path, target_size=(224, 224), batch_size=30))
# test_batches = (ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input).
#                 flow_from_directory(test_path, target_size=(224, 224), batch_size=30))

# Train the model
# model.fit(
#     train_generator,
#     steps_per_epoch=num_train_samples // batch_size,
#     epochs=1,
#     validation_data=valid_generator,
#     validation_steps=num_valid_samples // batch_size)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator, steps=num_test_samples // batch_size)
print('Test accuracy:', test_acc)
