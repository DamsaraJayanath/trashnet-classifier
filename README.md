# TrashNet Waste Classification using Deep Learning

## Overview
This project is an image classification system that identifies different types of waste using Deep Learning and Computer Vision. The model was trained on the TrashNet dataset using a custom Convolutional Neural Network (CNN) built with PyTorch.

The goal of the project is to support smart recycling and automated waste sorting systems.


## Classes
The model classifies waste into the following categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

## Project Structure


projects/<br>
├── data_setup.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Data loading and preprocessing)<br>
├── model_builder.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(CNN model architecture)<br>
├── engine.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Training and evaluation loops)<br>
├── train.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Main training script)<br>
├── utils.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Helper functions save, load, plot, etc.)<br>
├── predict.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Inference prediction script)<br>
├── models/<br>
│   └── best_model.pth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Saved trained model)<br>


## Tech Stack
- Python 
- PyTorch 
- Torchvision
- NumPy
- Matplotlib



## Model Performance
- Best Test Accuracy: **~75%**
- Dataset: TrashNet
- Input Image Size: 224 × 224



## How to Run

1️. Clone repository
```bash
git clone https://github.com/your-username/trashnet-classifier.git
cd trashnet-classifier
```
<br>

2️. Install dependencies
```bash
pip install torch torchvision matplotlib numpy
```

3️. Train model
```bash
python -m projects.train
```
4️. Run prediction
```bash
from projects.predict import predict_image
```
Example Output
```bash
Input image → waste item
Output → predicted class + confidence score
```

Key Learnings
- CNN architecture design
- Image augmentation techniques
- Training optimization in PyTorch
- Modular ML project structure
- Model evaluation and saving strategies
<br>

Future Improvements
- Use Transfer Learning (ResNet/EfficientNet) for higher accuracy
- Deploy using Streamlit or Gradio web app
- Add real-time webcam detection
- Improve dataset balancing

