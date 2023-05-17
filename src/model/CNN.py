from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def CNN(num_classes=10):
    model = Sequential()

    model.add(
        Conv2D(
            32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(128, 128, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model
