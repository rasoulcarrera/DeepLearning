from tensorflow.keras import layers, models


def inception_module(x, filters):
    """Inception module."""
    conv_1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    conv_3x3 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv_5x5)

    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    return layers.concatenate([conv_1x1, conv_3x3, conv_5x5, branch_pool], axis=-1)


def googLeNet(input_shape=(224, 224, 3), classes=1000):
    """Instantiates the GoogLeNet (Inception v1) architecture."""
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(img_input)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = inception_module(x, [112, 144, 288, 32, 64, 64])
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = inception_module(x, [384, 192, 384, 48, 128, 128])
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(classes, activation='softmax')(x)

    # Create model.
    model = models.Model(img_input, x, name='googLeNet')

    return model


model = googLeNet()
model.summary()
