import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# Define training datasets with standard augmentations and normalization
training_data = ImageFolder(
    root="data/training-data/",
    transform=v2.Compose([
        # Applying augumentation to increase the dataset size and to prevent overfitting
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),                  # Flip horizontally 50% of the time
        v2.RandomVerticalFlip(p=0.5),                    # Flip vertically 50% of the time
        v2.RandomRotation(degrees=30),
        v2.ToImage(),                              # Convert to PyTorch Image Tensor
        v2.ToDtype(torch.float32, scale=True),     # Scale integers [0, 255] to decimals [0.0, 1.0]
        # Normalized using standard ImageNet mean and standard deviation values
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
)

# Define validation datasets with deterministic reshaping and normalization
validation_data = ImageFolder(
    root="data/validation-data/",
    transform=v2.Compose([
        v2.Resize(256),                            # Shrink image preserving aspect ratio
        v2.CenterCrop(224),                        # Extract a clean center square
        v2.ToImage(),                              # Convert to PyTorch Image Tensor
        v2.ToDtype(torch.float32, scale=True),     # Scale integers [0, 255] to decimals [0.0, 1.0]
        # Normalized using standard ImageNet mean and standard deviation values
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
)



train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

