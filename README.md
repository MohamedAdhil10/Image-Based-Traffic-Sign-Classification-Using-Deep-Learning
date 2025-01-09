# Image-Based Traffic Sign Classification Using Deep Learning

## **Overview**
This project implements a deep learning model for traffic sign classification using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The goal is to build a robust model that can classify traffic signs into 43 categories, which can assist autonomous vehicles, traffic monitoring systems, and driver assistance systems in recognizing and interpreting traffic signs.

---

## **Project Components**

### **1. Data Loading and Preprocessing**
- The project uses `Train.csv` and `Test.csv` files to load the image paths and labels.
- Preprocessing steps include:
  - Resizing all images to a uniform size of `64x64` pixels.
  - Normalizing pixel values to the range `[0, 1]` for faster training convergence.
  - Splitting the training data into 80% training and 20% validation sets.
  - Converting class labels into one-hot encoded format using TensorFlow's `to_categorical`.

---

### **2. Model Building**
- A Convolutional Neural Network (CNN) was built using TensorFlow and Keras.
- Model architecture includes:
  - **2 Convolutional Layers**: Extract spatial features.
  - **MaxPooling Layers**: Reduce dimensionality while retaining important information.
  - **Dropout Layers**: Prevent overfitting by randomly dropping connections.
  - **Dense Layers**: Perform final classification using the `softmax` activation function.

---

### **3. Model Training**
- The model is trained for **15 epochs** with a batch size of **32**.
- The training process uses:
  - **Categorical Crossentropy Loss**: Suitable for multi-class classification.
  - **Adam Optimizer**: Adaptive learning rate optimization.
  - **Accuracy Metric**: Tracks training and validation accuracy during training.

---

### **4. Evaluation**
- The trained model is evaluated on the test dataset using the following metrics:
  - **Test Accuracy**: Percentage of correctly classified test samples.
  - **Test Loss**: Error on the test dataset.
- Additionally:
  - **Classification Report**: Includes precision, recall, F1-score, and support for each class.
  - **Confusion Matrix**: Displays misclassification patterns across all classes.

---

### **5. Visualization**
- **Accuracy and Loss Plots**:
  - Visualize training and validation accuracy and loss over the epochs.
- **Confusion Matrix Heatmap**:
  - Uses `seaborn` to plot a heatmap for the confusion matrix, highlighting areas of misclassification.

---

### **6. Model Saving**
- The trained model is saved as `traffic_sign_model.h5` for reuse.

---

## **Project Deliverables**
1. **Code Implementation**:
   - All functionality is implemented in the Jupyter Notebook `Image Based - DL.ipynb`.
2. **Trained Model**:
   - Saved as `traffic_sign_model.h5`.
3. **Evaluation Metrics**:
   - Accuracy and loss plots.
   - Classification report and confusion matrix.

---



## **Acknowledgments**
- Dataset: [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Tools: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn

## **Author**
Mohamed Adhil
