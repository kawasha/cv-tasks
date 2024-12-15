import os
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tf_keras.models import Sequential, load_model
from tf_keras.preprocessing.image import ImageDataGenerator

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 20  
train_path = 'dataset/train'
validation_path = 'dataset/validation'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  # Increased rotation range
    width_shift_range=0.3,  # More significant width shift
    height_shift_range=0.3,  # More significant height shift
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'  # Fill missing pixels after transformations
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

validation_data = validation_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define file paths for the models
cnn_model_path = 'cnn_model.h5'
knn_model_path = 'knn_model.pkl'

# Check if the CNN model exists and load it, else train it
if os.path.exists(cnn_model_path):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_path)
else:
    print("Training CNN model...")
    # Define and train the CNN model (as in your original code)
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, epochs=epochs, validation_data=validation_data)
    
    # Save the trained CNN model
    cnn_model.save(cnn_model_path)
    print("CNN model saved.")


# Function to extract features using the CNN model
def extract_features(model, data):
    features = []
    labels = []
    for _ in range(len(data)):
        imgs, lbls = next(data)
        feature_vectors = model.predict(imgs)  # Get feature vectors from CNN layers
        features.extend(feature_vectors)
        labels.extend(lbls)
    return np.array(features), np.array(labels)

# Extract features from training data
train_features, train_labels = extract_features(cnn_model, train_data)

# Extract features from validation data
validation_features, validation_labels = extract_features(cnn_model, validation_data)

# Check if the KNN model exists and load it, else train it
if os.path.exists(knn_model_path):
    print("Loading pre-trained KNN model...")
    knn = joblib.load(knn_model_path)
else:
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
    knn.fit(train_features, train_labels)
    
    # Save the trained KNN model
    joblib.dump(knn, knn_model_path)
    print("KNN model saved.")

# Evaluate the best KNN model on validation data
y_pred = knn.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"KNN model accuracy : {accuracy * 100:.2f}%")


# Select 10 random indices from the validation data
random_indices = random.sample(range(len(validation_labels)), 10)

# Get the corresponding images and labels
sample_images = []
real_labels = []
predicted_labels = []

# To retrieve individual images from the validation generator
validation_data.reset()  # Reset the generator to ensure we start from the beginning
for idx in random_indices:
    # Get the batch from the validation data
    imgs, lbls = next(validation_data)  # This returns a batch of images and labels
    
    # Since we are sampling, we want to access the individual image in the batch
    sample_images.append(imgs[idx % batch_size])  # Use modulo to get the correct image in the batch
    real_labels.append(lbls[idx % batch_size])  # Same for labels
    predicted_labels.append(y_pred[idx])  # Get the predicted label

# Plot the images with real and predicted labels
plt.figure(figsize=(12, 8))
for i, (image, real, predicted) in enumerate(zip(sample_images, real_labels, predicted_labels)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)  # No need for [0] as we're directly accessing the image
    plt.title(f"Real: {'Cat' if real == 0 else 'Dog'}\nPred: {'Cat' if predicted == 0 else 'Dog'}")
    plt.axis('off')

plt.tight_layout()
plt.show()