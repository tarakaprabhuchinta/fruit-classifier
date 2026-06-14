import torch
from torchvision.transforms import v2
from PIL import Image
from src.model import NeuralNetwork

# Class names in exact order as ImageFolder assigns them
class_names = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']

# Preprocess image — same as validation transform
transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = NeuralNetwork()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted_class].item()
    return class_names[predicted_class], confidence