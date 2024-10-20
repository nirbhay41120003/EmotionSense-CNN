# Import required packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory  # Updated import for loading images
import cv2

# Preprocess all train images
train_dataset = image_dataset_from_directory(
        'data/train',
        image_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        label_mode='categorical'  # Updated from class_mode to label_mode
)

# Preprocess all validation images
validation_dataset = image_dataset_from_directory(
        'data/test',
        image_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        label_mode='categorical'  # Updated from class_mode to label_mode
)

# Create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Disable OpenCL usage for OpenCV
cv2.ocl.setUseOpenCL(False)

# Compile the model
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_dataset,  # Use the new dataset directly
        epochs=50,
        validation_data=validation_dataset  # Use the new validation dataset directly
)

# Save the entire model (architecture + weights) in a single file
emotion_model.save('emotion_model.h5')

# Optionally, you can also save the model architecture separately (in JSON format)
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)
