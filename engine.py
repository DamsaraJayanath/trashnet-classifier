
import torch
from tqdm.auto import tqdm
from pathlib import Path



# Train Step

def train_step(model,
               dataloader,
               loss_fn,
               optimizer,
               device):

    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        train_loss += loss.item()
        preds = y_pred.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = train_loss / len(dataloader)
    train_acc = correct / total

    return train_loss, train_acc


# Test Step
def test_step(model,
              dataloader,
              loss_fn,
              device):

    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            loss = loss_fn(test_pred, y)

            test_loss += loss.item()

            preds = test_pred.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_loss = test_loss / len(dataloader)
    test_acc = correct / total

    return test_loss, test_acc

# TRAIN FUNCTION (Train loop + Test loop)
def train(model,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn,
          device,
          epochs,
          model_path="models/best_model.pth"):

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # create model directory
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_test_acc = 0.0

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_path)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    print(f"Model saved at: {model_path}")

    return results
