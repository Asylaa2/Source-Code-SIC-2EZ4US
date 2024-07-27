import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Tentukan direktori dataset
train_dir = 'DATABASE'
validation_dir = 'DATABASE'

# Preprocessing data
train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Verifikasi jumlah sampel di direktori training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Verifikasi jumlah sampel di direktori validation
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Tambahkan print statement untuk memeriksa jumlah sampel yang dimuat
print(f'Found {train_generator.samples} training samples across {train_generator.num_classes} classes.')
print(f'Found {validation_generator.samples} validation samples across {validation_generator.num_classes} classes.')

# Membuat model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # jumlah kelas sesuai dengan jumlah subdirektori di folder train
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Menyimpan model
model.save('asl_model.h5')
