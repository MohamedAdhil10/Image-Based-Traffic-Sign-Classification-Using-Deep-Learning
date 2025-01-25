# 🚦 Image-Based Traffic Sign Classification Using Deep Learning

This project implements a deep learning model for **traffic sign classification** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The system leverages **transfer learning** with a pre-trained **MobileNetV2** model, fine-tuned to classify 43 different traffic sign categories. The trained model can be deployed to predict traffic signs from new images through an interactive **Streamlit** application.

---

## 📀 Project Overview

Traffic signs play a crucial role in ensuring road safety. Automating traffic sign recognition enhances the functionality of autonomous vehicles and driver-assistance systems. This project builds an end-to-end pipeline that:
1. Preprocesses and augments the GTSRB dataset.
2. Trains a traffic sign classification model using **MobileNetV2**.
3. Evaluates the model on unseen data.
4. Deploys the trained model using a **Streamlit** web application for real-time traffic sign predictions.

---

## 🚀 Key Features

- **Transfer Learning**: MobileNetV2 as the feature extractor for efficiency and accuracy.
- **Data Augmentation**: Enhances robustness with rotations, brightness changes, and shifts.
- **Model Fine-Tuning**: Further trains deeper layers of MobileNetV2 for traffic sign-specific features.
- **Metrics Visualization**: Tracks accuracy and loss during training with clear visualizations.
- **Streamlit Deployment**: An interactive web app for traffic sign classification.

---

## 📂 Project Structure

```
.
├── Train.csv                # Training dataset (image paths and labels)
├── Test.csv                 # Testing dataset (image paths and labels)
├── traffic_sign_model_finetuned.h5 # Trained model saved for deployment
├── app.py                   # Streamlit app for predictions
├── model_training.py        # Main script for training and fine-tuning the model
├── README.md                # Project documentation
```

---

## 📊 Dataset

The **German Traffic Sign Recognition Benchmark (GTSRB)** dataset is used. It contains:
- **50,000+ images** of traffic signs.
- **43 classes**, including stop signs, speed limits, pedestrian crossings, and more.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Download the Dataset
- Download the **GTSRB dataset** from [Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
- Place `Train.csv` and `Test.csv` in the project directory.

### 3. Train the Model
Run the `model_training.py` script to train the model:
```bash
python model_training.py
```
- The trained model will be saved as `traffic_sign_model_finetuned.h5`.

### 4. Launch the Streamlit App
Use the `app.py` script to deploy the Streamlit web app:
```bash
streamlit run app.py
```
- Upload an image of a traffic sign, and the app will classify it.

---

## 📚 Code Explanation

### **1. Model Training**
- **Transfer Learning**: MobileNetV2 is used as a pre-trained base model.
- **Custom Layers**: Added layers for traffic sign classification:
  - `GlobalAveragePooling2D`: Reduces features into a compact vector.
  - `Dense`: Fully connected layers to learn traffic sign-specific patterns.
  - `Dropout`: Prevents overfitting.
  - `Softmax`: Outputs probabilities for 43 classes.
- **Optimization**: Adam optimizer with learning rate scheduling.

### **2. Data Augmentation**
- Applied real-time augmentation:
  - Rotation (±15°)
  - Brightness changes (80%-120%)
  - Shifts (10% horizontally and vertically)
  - Zooming (10%)

### **3. Evaluation**
- Accuracy, loss, precision, recall, and F1-score metrics.
- Confusion matrix for detailed class-wise performance.

---

## 🎯 Results

- **Validation Accuracy**: Achieved >95% accuracy during validation.
- **Test Accuracy**: Model generalizes well with high test accuracy.
- **Confusion Matrix**: Visualized model performance for all 43 traffic sign classes.

---

## 🖥️ Streamlit Web App

The Streamlit app allows users to:
1. Upload an image of a traffic sign.
2. View the predicted traffic sign class and confidence score.

---

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.10 or higher
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Pillow


---

## 📧 Contact

For any questions or suggestions, please reach out:
- **Email**: adhilm9991@gmail.com
- **GitHub**:https://github.com/MohamedAdhil10

---
