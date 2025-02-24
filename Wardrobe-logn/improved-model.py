import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_improved_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_improved_model():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    model = create_improved_model()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_fashion_model.h5',
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks
    )

    return model, history


if __name__ == "__main__":
    model, history = train_improved_model()
    model.save('improved_fashion_model.h5')
    print("Model saved successfully as .h5!")