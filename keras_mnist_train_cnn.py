import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras

# Load the Mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess and normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data to have a single channel (grayscale)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the Sequential model
model = Sequential([

    # Input layer
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    # Second convolutional layer
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten the output, flatten into 1 dim layer
    Flatten(),
    # Optional Dropout layer
    Dropout(0.5),
    # Fully connected layer
    Dense(10, activation='softmax')
])
# Print summary of the model
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Start training
history = model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.2)

# Evaluate the model with the test dataset
score = model.evaluate(x_test, y_test, verbose=0)

# Print the test accuracy
print(f'Test accuracy: {score[1] * 100:.2f}%')

# Select a random sample from the test dataset (Task 6)
random_index = np.random.randint(0, len(x_test))
sample_image = x_test[random_index].reshape(1, 28, 28, 1)

# Predict the digit
predicted_digit = model.predict(sample_image)[0]

# Display the digit image
plt.imshow(x_test[random_index].reshape(28, 28), cmap='gray')
plt.title(f'Actual Digit: {np.argmax(y_test[random_index])}\nPredicted Digit:')  # {predicted_digit}')
plt.show()

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
