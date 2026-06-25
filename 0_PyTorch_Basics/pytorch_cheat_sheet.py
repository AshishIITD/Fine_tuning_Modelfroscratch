# -*- coding: utf-8 -*-
"""
Module 0: Mastering PyTorch Basics (Your Guide to Writing Custom Models)

To write custom neural networks (like RNNs, LSTMs, Transformers, and LoRA) by yourself, 
you must master the core building blocks of PyTorch.

This script is an interactive, fully runnable guide that explains the 6 fundamental 
concepts you need to know. Run this file and read the terminal outputs to learn!

Table of Contents:
1. Tensors & Shapes (The fundamental data structure)
2. Tensor Manipulation (Reshaping, Transposing, Squeezing, Chunking)
3. Matrix Mathematics (Element-wise vs. Matrix Multiplication)
4. Autograd & Gradients (How PyTorch learns)
5. The nn.Module Blueprint (How to write a custom layer from scratch)
6. The 5-Step Training Loop Recipe (The training blueprint)
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("==================================================")
print("📚 MASTERING PYTORCH: AN INTERACTIVE GUIDE")
print("==================================================\n")

# =====================================================================
# CONCEPT 1: Tensors & Shapes
# =====================================================================
print("--------------------------------------------------")
print("CONCEPT 1: Tensors & Shapes")
print("--------------------------------------------------")
# A tensor is a multi-dimensional array (like a NumPy array) that can run on GPUs/TPUs.
# Every tensor has a shape (dimensions) and a data type (dtype).

# Creating tensors
x = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])

print(f"Tensor x:\n{x}")
print(f"Shape of x: {x.shape} (2 rows, 3 columns)")
print(f"Data type of x: {x.dtype}")
print(f"Device of x: {x.device} (Running on CPU by default)")

# Common ways to initialize tensors:
zeros = torch.zeros(2, 3)          # Tensor filled with 0s
ones = torch.ones(2, 3)            # Tensor filled with 1s
rand = torch.randn(2, 3)           # Tensor filled with random normal distribution (mean=0, std=1)
arange = torch.arange(0, 10, 2)    # [0, 2, 4, 6, 8]

print(f"Random Tensor:\n{rand}\n")


# =====================================================================
# CONCEPT 2: Tensor Manipulation (Crucial for Attention & Gates)
# =====================================================================
print("--------------------------------------------------")
print("CONCEPT 2: Tensor Manipulation")
print("--------------------------------------------------")
# Deep learning is all about reshaping tensors so they can be multiplied together.

t = torch.arange(12) # [0, 1, 2, ..., 11]
print(f"Original 1D tensor: {t} | Shape: {t.shape}")

# 1. Reshaping (.view or .reshape)
# Changes the dimensions without changing the underlying data.
t_2d = t.view(3, 4)
print(f"Reshaped to 2D (3x4):\n{t_2d} | Shape: {t_2d.shape}")

# Using -1: PyTorch automatically calculates the dimension for -1
t_3d = t.view(2, 3, -1) 
print(f"Reshaped to 3D (2x3x2):\n{t_3d} | Shape: {t_3d.shape}")

# 2. Transposing & Permuting
# Swapping dimensions. Crucial for Multi-Head Attention!
# .transpose(dim1, dim2) swaps two dimensions.
t_transposed = t_2d.transpose(0, 1) # Swaps rows and columns
print(f"Transposed (4x3):\n{t_transposed} | Shape: {t_transposed.shape}")

# 3. Adding/Removing Dimensions (.unsqueeze and .squeeze)
# .unsqueeze(dim) adds a dimension of size 1 at the specified index.
# .squeeze() removes all dimensions of size 1.
v = torch.tensor([1, 2, 3]) # shape: [3]
v_unsqueezed = v.unsqueeze(0) # shape: [1, 3] (adds a batch dimension!)
print(f"Unsqueezed shape: {v_unsqueezed.shape}")
print(f"Squeezed back shape: {v_unsqueezed.squeeze().shape}")

# 4. Chunking & Splitting (Used to split Q, K, V or LSTM gates)
# .chunk(num_chunks, dim) splits a tensor into equal parts along a dimension.
gate_inputs = torch.randn(1, 8) # Imagine 4 combined gates, each size 2
gate1, gate2, gate3, gate4 = torch.chunk(gate_inputs, 4, dim=1)
print(f"Chunked 1x8 tensor into 4 parts. Part 1 shape: {gate1.shape}\n")


# =====================================================================
# CONCEPT 3: Matrix Mathematics
# =====================================================================
print("--------------------------------------------------")
print("CONCEPT 3: Matrix Mathematics")
print("--------------------------------------------------")
# There is a massive difference between element-wise multiplication and matrix multiplication!

a = torch.tensor([[1.0, 2.0], 
                  [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], 
                  [7.0, 8.0]])

# 1. Element-wise multiplication (*)
# Multiplies elements at matching positions. Shapes must match.
element_wise = a * b
print(f"Element-wise multiplication (a * b):\n{element_wise}")

# 2. Matrix Multiplication (@ or torch.matmul)
# Performs standard dot-product matrix multiplication.
# Shape rule: [M, N] @ [N, P] = [M, P]
matrix_product = a @ b # or torch.matmul(a, b)
print(f"Matrix Multiplication (a @ b):\n{matrix_product}\n")


# =====================================================================
# CONCEPT 4: Autograd & Gradients (How PyTorch Learns)
# =====================================================================
print("--------------------------------------------------")
print("CONCEPT 4: Autograd & Gradients")
print("--------------------------------------------------")
# PyTorch tracks all operations on tensors that have `requires_grad=True`.
# This allows it to automatically compute derivatives (gradients) using backpropagation!

# Let's define a weight w, an input x, and a bias b
w = torch.tensor([3.0], requires_grad=True)
x = torch.tensor([2.0])
b = torch.tensor([1.0], requires_grad=True)

# Forward pass: y = w * x + b
y = w * x + b # y = 3 * 2 + 1 = 7

# Backward pass: compute derivatives
# This calculates dy/dw and dy/db automatically!
y.backward()

# Inspect gradients:
# dy/dw = x = 2.0
# dy/db = 1.0
print(f"Forward output y: {y.item()}")
print(f"Gradient of w (dy/dw): {w.grad.item()} (Should be equal to x = 2.0)")
print(f"Gradient of b (dy/db): {b.grad.item()} (Should be equal to 1.0)\n")


# =====================================================================
# CONCEPT 5: The nn.Module Blueprint (Writing a Custom Layer)
# =====================================================================
print("--------------------------------------------------")
print("CONCEPT 5: The nn.Module Blueprint")
print("--------------------------------------------------")
# In PyTorch, every neural network layer or model inherits from `nn.Module`.
# You must implement two functions:
# 1. `__init__`: Define your trainable weights (parameters) and sub-layers.
# 2. `forward`: Define the mathematical forward pass.

class MyLinearLayer(nn.Module):
    """
    A custom linear layer (y = x @ W + b) built completely from scratch!
    """
    def __init__(self, in_features, out_features):
        super(MyLinearLayer, self).__init__()
        # We wrap our weight tensor in nn.Parameter so PyTorch knows it is a 
        # trainable weight and automatically includes it in model.parameters().
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1, out_features))

    def forward(self, x):
        # x shape: [batch_size, in_features]
        # output shape: [batch_size, out_features]
        return torch.matmul(x, self.weight) + self.bias

# Instantiate our custom layer
my_layer = MyLinearLayer(in_features=3, out_features=2)
print("Trainable parameters in our custom layer:")
for name, param in my_layer.named_parameters():
    print(f"-> Parameter name: '{name}' | Shape: {param.shape}")

# Test forward pass
input_data = torch.randn(1, 3) # Batch size 1, 3 features
output_data = my_layer(input_data)
print(f"Input: {input_data} -> Output of custom layer: {output_data}\n")


# =====================================================================
# CONCEPT 6: The 5-Step Training Loop Recipe
# =====================================================================
print("--------------------------------------------------")
print("CONCEPT 6: The 5-Step Training Loop Recipe")
print("--------------------------------------------------")
# This is the exact code structure used to train 99% of PyTorch models.
# Learn this recipe by heart!

# 1. Create dummy dataset
inputs = torch.randn(10, 3)       # 10 samples, 3 features
targets = torch.randn(10, 2)      # 10 targets, 2 features

# 2. Initialize Model, Loss Function, and Optimizer
model = MyLinearLayer(in_features=3, out_features=2)
criterion = nn.MSELoss()          # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.1) # Stochastic Gradient Descent

print("Starting 3 epochs of training...")
for epoch in range(1, 4):
    # Step 1: Clear the previous gradients (always do this first!)
    optimizer.zero_grad()
    
    # Step 2: Forward pass (compute predictions)
    predictions = model(inputs)
    
    # Step 3: Compute the loss (how wrong the model is)
    loss = criterion(predictions, targets)
    
    # Step 4: Backward pass (compute gradients using backpropagation)
    loss.backward()
    
    # Step 5: Step the optimizer (update the weights based on the gradients)
    optimizer.step()
    
    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("\n==================================================")
print("🎉 CONGRATULATIONS! YOU HAVE MASTERED PYTORCH BASICS!")
print("==================================================")
print("You now know:")
print("1. How to create and inspect shapes of Tensors.")
print("2. How to reshape and transpose dimensions.")
print("3. How matrix multiplications work.")
print("4. How gradients flow through Autograd.")
print("5. How to structure a custom layer using nn.Module and nn.Parameter.")
print("6. How to write a standard training loop.")
print("You are ready to read and write custom LLM architectures!")
print("==================================================")
