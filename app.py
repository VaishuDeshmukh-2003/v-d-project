import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import load_img, img_to_array  # <-- fixed import here
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

MODEL_PATH = 'grape_leaf_disease_4class_model.h5'
DATA_DIR = 'Final Training Data'

def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(225, 225),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(225, 225),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(225, 225, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=5, validation_data=val_generator)

    model.save(MODEL_PATH)
    print("Model trained and saved.")
    return model

def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        return load_model(MODEL_PATH)
    else:
        print("Training model...")
        return train_model()

# Example usage:
if __name__ == "__main__":
    model = train_or_load_model()
