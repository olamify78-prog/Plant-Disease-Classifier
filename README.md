# Plant-Disease-Classifier # Smart Plant Disease Diagnostics 🌱

An AI-driven image classification system built using **TensorFlow** and **Transfer Learning** to identify agricultural crop diseases from leaf images.

## 🚀 How It Works
This project leverages **MobileNetV2** (pre-trained on the ImageNet dataset). The base feature-extraction layers are frozen, and a custom dense neural network head is trained on plant datasets to recognize specific blights and spots.

## 📊 Dataset
The model is trained using the public **New Plant Diseases Dataset** from Kaggle, which features over 87,000 images of healthy and infected crop leaves.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Libraries:** NumPy, Pandas
