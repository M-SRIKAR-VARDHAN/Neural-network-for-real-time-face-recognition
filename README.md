# Face Recognition using Deep Learning

## 📌 Overview
This project implements a deep learning-based face recognition system using a MobileNetV2 model. The system detects faces from an image or video feed and classifies them into predefined categories.

## 🚀 Features
- **Face Detection** using MTCNN
- **Face Recognition** using MobileNetV2
- **Dataset Augmentation** for improving model performance
- **Live Webcam Inference** to recognize faces in real time
- **Model Training & Validation** with jupyter notebook

---

## 📂 Dataset & Preprocessing
- The dataset consists of images categorized into folders by person names.
- **Data Augmentation** is applied to expand the dataset for better generalization.
- Preprocessing includes image resizing, normalization, and conversion to tensor format.

---

## ⚡ Installation
First, install the required dependencies:
```bash
pip install torch torchvision facenet-pytorch matplotlib opencv-python tqdm
```

---

## 🏗️ Model Architecture
- Uses a **pretrained MobileNetV2** model for feature extraction.
- The **classifier layer** is modified to match the number of classes.
- The model is trained using **CrossEntropyLoss** and **Adam optimizer**.

---

## 📊 Training the Model

The training script includes early stopping, learning rate scheduling, and validation metrics.

---

## 🎥 Live Face Recognition
To test the trained model in real-time using a webcam, run:
```bash
run the ipynb file to train and load the model for use
```
Press `Q` to exit the webcam window.
![Description](image.jpg)

---

## 📝 Future Improvements
- Implement support for **unknown faces**
- Enhance the model with **transformer-based architectures**
- Deploy as a **web or mobile application**

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

### 🔗 Connect with Me
[LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/srikar-vardhan/)) | [GitHub]([https://github.com/](https://github.com/M-SRIKAR-VARDHAN))
