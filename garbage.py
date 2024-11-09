import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Set the dataset path
data_dir = '/Users/raushankuamar/Desktop/Garbage/dataset/Garbage classification'  # Update this path to your dataset

# Load the dataset
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',  # Use 'binary' for binary classification
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',  # Use 'binary' for binary classification
        subset='validation'
    )

    return train_generator, val_generator

# Define the CNN model
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Use 'sigmoid' for binary classification
    ])
    return model

# Train the model
def train_model(model, train_gen, val_gen):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=10)
    return history

# Real-time image classification from webcam
def classify_from_camera(model, class_names):
    cap = cv2.VideoCapture(0)  # Start the webcam
    
    while True:
        ret, frame = cap.read()  # Capture a frame
        if not ret:
            break  # Break the loop if no frame is captured
        
        # Preprocess the image
        img = cv2.resize(frame, (150, 150))  # Resize the frame
        img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize the image
        
        # Make prediction
        predictions = model.predict(img_array)  # Get model predictions
        predicted_class_index = np.argmax(predictions[0])  # Get the index of the predicted class
        predicted_class_name = class_names[predicted_class_index]  # Get the actual class name
        
        # Display prediction on frame
        cv2.putText(frame, f'Predicted: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)  # Show the frame with the prediction
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed
    
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    # Load data
    train_gen, val_gen = load_data(data_dir)

    # Get the number of classes
    num_classes = len(train_gen.class_indices)
    print(f"Classes: {train_gen.class_indices}")

    # Create and train the model
    model = create_model(num_classes)
    history = train_model(model, train_gen, val_gen)

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Get class names from the generator
    class_names = list(train_gen.class_indices.keys())

    # Start real-time classification
    classify_from_camera(model, class_names)
