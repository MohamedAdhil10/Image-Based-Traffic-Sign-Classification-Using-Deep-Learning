import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (64, 64)  
NUM_CLASSES = 43     
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 5
TOTAL_EPOCHS = EPOCHS_INITIAL + EPOCHS_FINE_TUNE

# 1. Data Generators
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2  # Split data into training and validation sets
)

# Load data dynamically with `flow_from_dataframe`
train_csv = pd.read_csv('Train.csv')
test_csv = pd.read_csv('Test.csv')

train_csv['ClassId'] = train_csv['ClassId'].astype(str)
test_csv['ClassId'] = test_csv['ClassId'].astype(str)

# Training data generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Test data generator (without augmentation)
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 2. Model Building
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze base model layers initially

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.6),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Initial Training
history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INITIAL,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# 4. Fine-Tuning
base_model.trainable = True  # Unfreeze the base model
fine_tune_at = len(base_model.layers) // 2
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tuning phase
history_fine_tune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=EPOCHS_INITIAL,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# 5. Model Evaluation
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# 6. Metrics Visualization
def plot_metrics(history_initial, history_fine_tune):
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history_initial.history['accuracy'], label='Train Accuracy (Initial)')
    plt.plot(history_initial.history['val_accuracy'], label='Validation Accuracy (Initial)')
    plt.plot(history_fine_tune.history['accuracy'], label='Train Accuracy (Fine-Tune)')
    plt.plot(history_fine_tune.history['val_accuracy'], label='Validation Accuracy (Fine-Tune)')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history_initial.history['loss'], label='Train Loss (Initial)')
    plt.plot(history_initial.history['val_loss'], label='Validation Loss (Initial)')
    plt.plot(history_fine_tune.history['loss'], label='Train Loss (Fine-Tune)')
    plt.plot(history_fine_tune.history['val_loss'], label='Validation Loss (Fine-Tune)')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_metrics(history_initial, history_fine_tune)

# 7. Prediction and Confusion Matrix
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=[str(i) for i in range(NUM_CLASSES)]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(NUM_CLASSES)], yticklabels=[str(i) for i in range(NUM_CLASSES)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 8. Save the Model
model.save('traffic_sign_model_finetuned.h5')
print("Model saved as 'traffic_sign_model_finetuned.h5'.")
