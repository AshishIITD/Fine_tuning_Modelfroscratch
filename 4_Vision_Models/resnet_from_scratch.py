# -*- coding: utf-8 -*-
"""
Module 4a: Mini-ResNet from Scratch (Residual CNN)

In this file, we implement a Gated/Residual Convolutional Neural Network (ResNet) 
from scratch in PyTorch, explaining Conv2d, BatchNorm2d, MaxPool2d, and the legendary 
Skip Connection (Residual link) that revolutionized computer vision.

To make this completely runnable offline, we generate a synthetic image dataset 
of 3 geometric shapes: Crosses (X), Circles (O), and Squares ([]).

Formulas:
    Residual Connection: y = ReLU( F(x) + x )
    Where F(x) is the output of convolutional layers, and x is the identity skip connection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# SECTION 1: Generating Synthetic Shape Dataset (28x28 grayscale)
# =====================================================================
def generate_shape_dataset(num_samples=150):
    """
    Generates synthetic 28x28 grayscale images representing:
    Class 0: Cross (X)
    Class 1: Circle (O)
    Class 2: Square ([])
    """
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Create blank black image (zeros)
        img = np.zeros((28, 28), dtype=np.float32)
        shape_type = np.random.randint(0, 3)
        
        # Add some random noise to the background
        img += np.random.normal(0, 0.05, (28, 28))
        
        # Draw shape with slight random offset (translation invariance test!)
        offset_y = np.random.randint(-2, 3)
        offset_x = np.random.randint(-2, 3)
        cy, cx = 14 + offset_y, 14 + offset_x
        
        if shape_type == 0:  # Cross (X)
            # Draw two intersecting diagonal lines
            for i in range(-7, 8):
                img[cy + i, cx + i] = 1.0
                img[cy + i, cx - i] = 1.0
                
        elif shape_type == 1:  # Circle (O)
            # Draw a circle using basic trigonometry
            for angle in np.linspace(0, 2 * np.pi, 30):
                y = int(cy + 6 * np.sin(angle))
                x = int(cx + 6 * np.cos(angle))
                img[y, x] = 1.0
                
        elif shape_type == 2:  # Square ([])
            # Draw boundaries of a square
            for i in range(-6, 7):
                img[cy - 6, cx + i] = 1.0  # Top edge
                img[cy + 6, cx + i] = 1.0  # Bottom edge
                img[cy + i, cx - 6] = 1.0  # Left edge
                img[cy + i, cx + 6] = 1.0  # Right edge
                
        # Clip to valid pixel boundaries [0.0, 1.0] and add channel dimension [1, 28, 28]
        img = np.clip(img, 0.0, 1.0)
        images.append(np.expand_dims(img, axis=0))
        labels.append(shape_type)
        
    return torch.tensor(images), torch.tensor(labels)


# =====================================================================
# SECTION 2: Residual Block Implementation
# =====================================================================
class ResidualBlock(nn.Module):
    """
    The fundamental building block of ResNet.
    Contains two Convolutional layers and a Skip Connection (Shortcut) 
    that adds the input directly to the output.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        # Conv 1: keeps shape same using padding=1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)  # Normalizes activations across batch
        self.relu = nn.ReLU(inplace=True)
        
        # Conv 2:
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Save the identity input (the skip connection highway!)
        residual = x
        
        # Pass through the convolutional path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # THE RESIDUAL MAGIC:
        # Add the original input (shortcut) back to the features!
        # This allows gradients to flow directly back during backpropagation,
        # preventing the vanishing gradient problem.
        out += residual
        
        # Final activation
        out = self.relu(out)
        return out


# =====================================================================
# SECTION 3: Mini-ResNet Architecture
# =====================================================================
class MiniResNet(nn.Module):
    """
    A lightweight Residual Convolutional Neural Network.
    """
    def __init__(self, num_classes=3):
        super(MiniResNet, self).__init__()
        
        # 1. Initial Convolution: maps 1 grayscale channel to 8 feature channels
        self.init_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Max Pooling: reduces spatial size from 28x28 to 14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Residual Blocks (Our deep feature extractors)
        self.res_block1 = ResidualBlock(channels=8)
        
        # 4. Another Convolution to increase channel depth, reducing size to 7x7
        self.downsample_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample_bn = nn.BatchNorm2d(16)
        
        self.res_block2 = ResidualBlock(channels=16)
        
        # 5. Fully Connected Classifier Head
        # After downsampling: size is 7x7 with 16 channels -> Flatten to 16 * 7 * 7 = 784 features
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        # Initial layers
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.relu(out)
        out = self.pool(out)  # Shape: [Batch, 8, 14, 14]
        
        # First residual block
        out = self.res_block1(out)  # Shape: [Batch, 8, 14, 14]
        
        # Downsample conv (reduces spatial resolution, increases depth)
        out = self.downsample_conv(out)
        out = self.downsample_bn(out)
        out = self.relu(out)  # Shape: [Batch, 16, 7, 7]
        
        # Second residual block
        out = self.res_block2(out)  # Shape: [Batch, 16, 7, 7]
        
        # Flatten and Classify
        out = out.view(out.size(0), -1)  # Flatten: [Batch, 784]
        logits = self.fc(out)            # Classifier: [Batch, 3]
        return logits


# =====================================================================
# SECTION 4: Dataset Loading and Training
# =====================================================================
if __name__ == "__main__":
    print("==================================================")
    print("🎨 INITIALIZING MINI-RESNET GEOMETRIC SHAPE CLASSIFIER")
    print("==================================================")
    
    # 1. Generate Datasets
    print("Generating synthetic dataset (Crosses, Circles, Squares)...")
    train_images, train_labels = generate_shape_dataset(num_samples=180)
    test_images, test_labels = generate_shape_dataset(num_samples=30)
    
    print(f"Train Dataset Shape: {train_images.shape} (180 images of 28x28 pixels)")
    print(f"Test Dataset Shape:  {test_images.shape} (30 images of 28x28 pixels)\n")
    
    # 2. Initialize Model, Loss, and Optimizer
    model = MiniResNet(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Mini-ResNet initialized with {total_params:,} parameters.")
    
    # 3. Training Loop
    print("\nTraining Mini-ResNet on shape classification...")
    epochs = 25
    batch_size = 30
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        # Simple batching
        permutation = torch.randperm(train_images.size(0))
        for i in range(0, train_images.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_images[indices], train_labels[indices]
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= train_images.size(0)
        
        # Evaluate accuracy on test set
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_images)
                predictions = torch.argmax(test_logits, dim=1)
                accuracy = (predictions == test_labels).float().mean().item() * 100
            print(f"Epoch {epoch:2d}/{epochs} | Average Train Loss: {epoch_loss:.4f} | Test Set Accuracy: {accuracy:.1f}%")
            
    # Final predictions check
    model.eval()
    with torch.no_grad():
        sample_logits = model(test_images[:5])
        sample_preds = torch.argmax(sample_logits, dim=1)
        
    class_map = {0: "Cross (X)", 1: "Circle (O)", 2: "Square ([])"}
    print("\n--- SAMPLE PREDICTIONS CHECK ---")
    for idx in range(5):
        actual = class_map[test_labels[idx].item()]
        predicted = class_map[sample_preds[idx].item()]
        status = "✅ Correct" if actual == predicted else "❌ Incorrect"
        print(f"Sample {idx+1} | Actual: {actual:<12} | Predicted: {predicted:<12} | {status}")
    print("---------------------------------")
