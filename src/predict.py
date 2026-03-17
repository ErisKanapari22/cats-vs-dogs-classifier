import sys
import torch
from PIL import Image
from torchvision import transforms

from src.model import CNNModel
from src.config import DEVICE, IMAGE_SIZE, MEAN, STD, MODEL_DIR


def load_model():
    model = CNNModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def predict(image_path):
    model = load_model()
    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # [1, 3, H, W]
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        label = "Dog"
        confidence = prob
    else:
        label = "Cat"
        confidence = 1 - prob

    print(f"\nPrediction: {label} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
