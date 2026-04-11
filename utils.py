
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. SET RANDOM SEEDS
def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 2. SAVE MODEL
def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / model_name
    torch.save(model.state_dict(), model_path)

    print(f"[INFO] Model saved to: {model_path}")

# 3. LOAD MODEL
def load_model(model: torch.nn.Module, model_path: str, device: str):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"[INFO] Model loaded from: {model_path}")
    return model


# 4. PLOT LOSS & ACCURACY
def plot_results(results: dict):
    epochs = range(len(results["train_loss"]))

    # Loss plot
    plt.figure()
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, results["train_acc"], label="Train Accuracy")
    plt.plot(epochs, results["test_acc"], label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

import time

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        end_time = time.time()
        elapsed = end_time - self.start_time
        print(f"[INFO] Time taken: {elapsed:.2f} seconds")
        return elapsed
