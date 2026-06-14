from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            # Feature extraction 1
            # Conv2d: slides 32 filters across image, each filter learns a different pattern
            # GroupNorm: normalizes within each image independently, stable for small datasets
            # ReLU: kills negative values, keeps only strong pattern detections
            # MaxPool2d: takes every 2×2 block and keeps max value, shrinks 224×224 to 112×112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 224 → 112

            # Feature extraction 2
            # Conv2d: slides 64 filters across 32 pattern maps, learns combinations of patterns
            # GroupNorm: normalizes 64 channel maps within each image
            # ReLU: kills weak detections, keeps strong ones
            # MaxPool2d: shrinks 112×112 to 56×56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 112 → 56

            # Feature extraction 3
            # Conv2d: slides 128 filters across 64 pattern maps, learns higher level features
            # GroupNorm: normalizes 128 channel maps within each image
            # ReLU: kills weak detections, keeps strong ones
            # MaxPool2d: shrinks 56×56 to 28×28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 56 → 28

            # Takes each of 128 maps to 4×4
            nn.AdaptiveAvgPool2d(4),      # 28 → 4×4

            # Classifier
            # Flatten: reshapes 128×4×4 3D maps into 2048 1D vector for Linear layer
            # Dropout: randomly kills 50% of inputs, prevents Linear layer from memorizing
            # Linear: learns which combination of patterns adds up to which fruit class
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 10)   # 128 channels now instead of 64
        )

    def forward(self, x):
        # Pass input through the entire stack and return raw scores (logits) for each class
        logits = self.conv_relu_stack(x)
        return logits