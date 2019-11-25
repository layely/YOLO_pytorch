import torch
from torch import nn

class YOLO(nn.Module):
    def __init__(self, image_shape, S, B, C):
        """
            image_shape: Shape of the input image
            S: SxS grid dimension
            B: Number of bounding boxes
            C: Number of classes
        """
        super(YOLO, self).__init__()

        # Keep these values, we will need them in forward pass.
        self.input_shape = image_shape
        self.S = S
        self.B = B
        self.C = C

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # Compute visual encoder output dim
        features_size = self.get_output_dim(self.feature_extractor, image_shape)

        # For each image, we predict S*S grids.
        # For each grid, B bounding boxes and C classes will be predicted.
        # Each bounding box is determined by 5 values: x, y, w, h, and confidence.
        # x, y represent the center of the bounding box relative to the image.
        # w, h represent the width and height of "" "" "".
        # Confidence represent how confident the network is about the prediction.
        output_size = S * S * (B * 5 + C)
        self.classifier = nn.Sequential(
            nn.Linear(features_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(output_size, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        y = self.classifier(features)
        return y.view(-1, self.S, self.B*5 + self.C)

    def get_output_dim(self, model, image_dim):
        return model(torch.rand(1, *(image_dim))).data.view(1, -1).size(1)