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


projects/<br>
├── data_setup.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Data loading and preprocessing)<br>
├── model_builder.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(CNN model architecture)<br>
├── engine.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Training and evaluation loops)<br>
├── train.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Main training script)<br>
├── utils.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Helper functions save, load, plot, etc.)<br>
├── predict.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Inference prediction script)<br>
├── models/<br>
│   └── best_model.pth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Saved trained model)<br>

---

## ⚙️ Tech Stack
- Python 
- PyTorch 
- Torchvision
- NumPy
- Matplotlib

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
```
<br>

2️⃣ Install dependencies
```bash
pip install torch torchvision matplotlib numpy
```

3️⃣ Train model
```bash
python -m projects.train
```
4️⃣ Run prediction
```bash
from projects.predict import predict_image
```
📷 Example Output
```bash
Input image → waste item
Output → predicted class + confidence score
```

💡 Key Learnings
- CNN architecture design
- Image augmentation techniques
- Training optimization in PyTorch
- Modular ML project structure
- Model evaluation and saving strategies
<br>

📌 Future Improvements
- Use Transfer Learning (ResNet/EfficientNet) for higher accuracy
- Deploy using Streamlit or Gradio web app
- Add real-time webcam detection
- Improve dataset balancing

