import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (96, 96)  
NUM_CLASSES = 43
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS_INITIAL = 15
EPOCHS_FINE_TUNE = 10
TOTAL_EPOCHS = EPOCHS_INITIAL + EPOCHS_FINE_TUNE

# Load dataset
train_csv = pd.read_csv('Train.csv')
test_csv = pd.read_csv('Test.csv')
train_csv['ClassId'] = train_csv['ClassId'].astype(str)
test_csv['ClassId'] = test_csv['ClassId'].astype(str)

# **Enhanced Data Augmentation**
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.6, 1.4],  # More brightness variation
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest',
    rescale=1.0 / 255.0,  # Normalization
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalize test images

# Training and Validation Generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Test Generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_csv['ClassId']),
    y=train_csv['ClassId']
)
class_weights = dict(enumerate(class_weights))

# **Base Model (MobileNetV2)**
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze initially

# **Custom Model Architecture**
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),  # Improve generalization
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

# **Compile Model**
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# **Learning Rate Scheduler & Early Stopping**
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# **Initial Training**
history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INITIAL,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stopping]
)

# **Fine-Tuning**
for layer in base_model.layers[-50:]:  # Unfreeze last 50 layers
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine_tune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=EPOCHS_INITIAL,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stopping]
)

# **Evaluate on Test Data**
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

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

# **Save Model**
model.save('traffic_sign_model_optimized.h5')
