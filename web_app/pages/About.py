import streamlit as st

st.set_page_config(page_title="About the Project", page_icon="📘")

st.title("📘 About the Project")
st.markdown("""
Welcome to the **Cat, Dog, and Wildlife Image Classifier**! 🐱🐶🦁  
This web app allows you to upload one or more images and classify them into one of the following categories:

- **Cat**
- **Dog**
- **Wildlife**

---

### 🎯 Project Objective

The goal of this project is to explore image classification using **deep learning**, particularly **Convolutional Neural Networks (CNNs)**, and apply them through a user-friendly web application using **Streamlit**.

---

### 🗂 Dataset

The dataset consists of labeled images belonging to three categories:
- Cats 🐱
- Dogs 🐶
- Wild Animals 🐯🦓🦍

Each image has been resized and normalized for consistency and fed into deep learning models for training and evaluation.

---

### 🧠 Model Architectures

Three models were trained and compared:

#### 1. 🧪 Custom CNN (Convolutional Neural Network)
A basic CNN built from scratch using PyTorch. It consists of:
- Two convolutional layers
- Max pooling
- Fully connected (dense) layers
- ReLU activation

This model helps understand how image features are learned without using pre-trained weights.

#### 2. 🏗️ ResNet50 (Transfer Learning)
ResNet50 is a **pre-trained deep neural network** developed by Microsoft. It uses **skip connections** (residuals) to improve gradient flow, enabling training of very deep networks.

We loaded a pre-trained ResNet50 (`torchvision.models.resnet50(pretrained=True)`) and fine-tuned the final fully connected layer to classify into 3 classes.

#### 3. ⚡ EfficientNetB0
EfficientNet is a **highly optimized model** developed by Google. It balances width, depth, and resolution efficiently.

EfficientNetB0 is the smallest variant and provides:
- **Faster inference**
- **Higher accuracy**
- **Fewer parameters**

We replaced its classifier layer to match our 3-class problem.

---

### 🛠️ Web App Inference

This **web application uses the EfficientNetB0 model** for all predictions and image classification tasks.

✅ EfficientNetB0 was chosen for:
- **High accuracy (99.8% validation)**
- **Efficient performance during inference**
- **Smaller model size suitable for web deployment**

You can upload one or multiple images and get predictions with:
- **Predicted class**
- **Confidence score**
- **Inference time**

---

### 🏆 Model Performance

| Model         | Validation Accuracy |
|---------------|----------------------|
| Custom CNN    | 95.4%                |
| ResNet50      | 99.5%                |
| EfficientNetB0| 99.8% ✅             |

---

### 🛠️ Tools & Libraries
- Python
- PyTorch
- Streamlit

---

### 🙌 Credits
Developed as part of a deep learning exploration project.  
If you'd like to see the source code or contribute, feel free to reach out or check the GitHub repo.

---

Have fun experimenting! 🧪🖼️
""")
