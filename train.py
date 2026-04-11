
import torch
from timeit import default_timer as timer

from projects.data_setup import create_dataloaders
from projects.model_builder import TrashNetV3
from projects.engine import train


# Set Hyperparameters
NUM_EPOCHS = 60
BATCH_SIZE = 32
IMG_SIZE = 224
HIDDEN_UNITS = 64
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Get dataloaders and classnames
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir="data/TrashNet/train",
    test_dir="data/TrashNet/test",
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE
)


# Model
model = TrashNetV3(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(DEVICE)

# Set Loss function and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
start_time = timer()

results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=DEVICE,
    epochs=NUM_EPOCHS,
    model_path="models/best_model.pth"
)

end_time = timer()

print(f"\nTotal training time: {end_time - start_time:.3f} seconds")
print(f"Best model saved to: models/best_model.pth")
