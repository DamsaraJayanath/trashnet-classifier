# ♻️ TrashNet Waste Classification using Deep Learning

## 📌 Overview
This project is a deep learning-based image classification system that classifies waste items into different categories using the **TrashNet dataset**.  
A custom Convolutional Neural Network (CNN) built with PyTorch is trained to recognize different types of waste for smart recycling applications.

---

## 🚀 Features
- Custom CNN built from scratch using PyTorch
- Data preprocessing and augmentation pipeline
- Train/Test split automation
- Model training with accuracy tracking
- Best model saving mechanism
- Inference system for real image prediction
- Modular and production-style project structure

---

## 🧠 Classes
The model classifies waste into the following categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## 🏗️ Project Structure


projects/
├── data_setup.py # Data loading and preprocessing
├── model_builder.py # CNN model architecture
├── engine.py # Training and evaluation loops
├── train.py # Main training script
├── utils.py # Helper functions (save, load, plot, etc.)
├── predict.py # Inference / prediction script

models/
└── best_model.pth # Saved trained model


---

## ⚙️ Tech Stack
- Python 🐍
- PyTorch 🔥
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn (optional utilities)

---

## 📊 Model Performance
- Best Test Accuracy: **~75%**
- Dataset: TrashNet
- Input Image Size: 224 × 224

---

## 🧪 How to Run

### 1️⃣ Clone repository
```bash
git clone https://github.com/your-username/trashnet-classifier.git
cd trashnet-classifier
2️⃣ Install dependencies
pip install torch torchvision matplotlib numpy
3️⃣ Train model
python -m projects.train
4️⃣ Run prediction
from projects.predict import predict_image
📷 Example Output
Input image → waste item
Output → predicted class + confidence score
💡 Key Learnings
CNN architecture design
Image augmentation techniques
Training optimization in PyTorch
Modular ML project structure
Model evaluation and saving strategies
📌 Future Improvements
Use Transfer Learning (ResNet/EfficientNet) for higher accuracy
Deploy using Streamlit or Gradio web app
Add real-time webcam detection
Improve dataset balancing
👨‍💻 Author

Damsara Jayanath
AI & Data Science Student
Aspiring AI Engineer 🚀
