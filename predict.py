
import random
from PIL import Image
import torch
import matplotlib.pyplot as plt


# Predict Function
def predict_image(model,
                  image_path_list,
                  transform,
                  class_names,
                  device,
                  n: int = 5):

    model.eval()

    # Get random images
    image_paths = random.sample(image_path_list, k=n)

    plt.figure(figsize=(12, 5 * n))

    for i, image_path in enumerate(image_paths):

        # load image
        image = Image.open(image_path).convert("RGB")
        true_class = image_path.parent.name

        # preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)

        # prediction
        with torch.inference_mode():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_prob, pred_class_idx = torch.max(probs, dim=1)

        pred_class = class_names[pred_class_idx.item()]
        confidence = pred_prob.item() * 100

        color = "green" if pred_class == true_class else "red"

        # plot original image
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f"True: {true_class}")
        plt.axis("off")

        # plot prediction
        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(image)
        plt.title(f"Pred: {pred_class}\nConfidence: {confidence:.1f}%",
                  color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
