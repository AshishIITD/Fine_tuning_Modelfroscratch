# -*- coding: utf-8 -*-
"""
Module 0.5: Mastering TensorFlow Basics (Your Guide to Writing Custom Models)

To be a complete deep learning engineer, you should know both PyTorch and TensorFlow.
This script is a fully commented guide showing how the 6 core concepts map to TensorFlow!

Key Structural Differences:
1. PyTorch tracks gradients implicitly on tensors with requires_grad=True.
2. TensorFlow tracks gradients explicitly using a 'tf.GradientTape()' context manager.
3. PyTorch uses 'nn.Module'; TensorFlow uses 'tf.keras.Model' or 'tf.keras.layers.Layer'.

Table of Contents:
1. Tensors, Constants, and Variables (tf.constant vs tf.Variable)
2. Tensor Manipulation (tf.reshape, tf.transpose, tf.split)
3. Matrix Mathematics (Element-wise vs. tf.matmul)
4. Autograd via GradientTape (How TensorFlow learns)
5. The tf.keras Model Blueprint (How to write custom layers)
6. The Custom Training Loop Recipe (The training blueprint)
"""

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow is not installed in this environment.")
    print("To run this script, please install it first by running:")
    print("   pip install tensorflow")
    print("Showing the educational code reference below:\n")
    # Define dummy placeholders so the script can still compile and show text
    class tf_dummy:
        def __init__(self):
            self.constant = lambda *args, **kwargs: None
            self.Variable = lambda *args, **kwargs: None
            self.reshape = lambda *args, **kwargs: None
            self.transpose = lambda *args, **kwargs: None
            self.matmul = lambda *args, **kwargs: None
    tf = tf_dummy()

import numpy as np

print("==================================================")
print(" MASTERING TENSORFLOW: AN INTERACTIVE GUIDE")
print("==================================================")

# =====================================================================
# CONCEPT 1: Tensors, Constants & Variables
# =====================================================================
print("\n--------------------------------------------------")
print("CONCEPT 1: Tensors, Constants & Variables")
print("--------------------------------------------------")
# In TensorFlow, there are two main types of tensors:
# 1. tf.constant: Immutable tensors (cannot be changed, like PyTorch tensors by default).
# 2. tf.Variable: Mutable tensors (can be updated, used for trainable weights).

# Creating a constant tensor (equivalent to torch.tensor)
x = tf.constant([[1.0, 2.0, 3.0], 
                 [4.0, 5.0, 6.0]], dtype=tf.float32)

print(f"Tensor x:\n{x}")
print(f"Shape of x: {x.shape} (Equivalent to PyTorch .shape)")
print(f"Data type of x: {x.dtype}")

# Creating a variable tensor (equivalent to PyTorch nn.Parameter)
w = tf.Variable([[0.5], [0.5], [0.5]], dtype=tf.float32, name="weights")
print(f"\nVariable w:\n{w}")
print(f"Is w trainable? Yes, variables are tracked for gradients by default.")


# =====================================================================
# CONCEPT 2: Tensor Manipulation (Reshaping & Dimensions)
# =====================================================================
print("\n--------------------------------------------------")
print("CONCEPT 2: Tensor Manipulation")
print("--------------------------------------------------")
# Reshaping and swapping dimensions works very similarly to PyTorch.

t = tf.range(12) # [0, 1, ..., 11] (Equivalent to torch.arange)
print(f"Original 1D tensor: {t}")

# 1. Reshaping (tf.reshape)
# Equivalent to PyTorch .view() or .reshape()
t_2d = tf.reshape(t, (3, 4))
print(f"Reshaped to 2D (3x4):\n{t_2d}")

# Using -1 for automatic dimension inference
t_3d = tf.reshape(t, (2, 3, -1))
print(f"Reshaped to 3D (2x3x2):\n{t_3d}")

# 2. Transposing (tf.transpose)
# Equivalent to PyTorch .transpose() or .permute()
t_transposed = tf.transpose(t_2d) # Swaps rows and columns
print(f"Transposed (4x3):\n{t_transposed}")

# 3. Adding/Removing Dimensions (tf.expand_dims & tf.squeeze)
# tf.expand_dims(t, axis) is equivalent to PyTorch .unsqueeze(dim)
# tf.squeeze(t) is equivalent to PyTorch .squeeze()
v = tf.constant([1, 2, 3])
v_expanded = tf.expand_dims(v, axis=0) # shape: [1, 3]
print(f"Expanded shape: {v_expanded.shape}")
print(f"Squeezed back shape: {tf.squeeze(v_expanded).shape}")

# 4. Splitting & Chunking
# tf.split(tensor, num_or_size_splits, axis) is equivalent to PyTorch .chunk()
gate_inputs = tf.random.normal((1, 8))
gate1, gate2, gate3, gate4 = tf.split(gate_inputs, num_or_size_splits=4, axis=1)
print(f"Split 1x8 tensor into 4 parts. Part 1 shape: {gate1.shape}")


# =====================================================================
# CONCEPT 3: Matrix Mathematics
# =====================================================================
print("\n--------------------------------------------------")
print("CONCEPT 3: Matrix Mathematics")
print("--------------------------------------------------")
# Element-wise vs Matrix Multiplication

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 1. Element-wise multiplication (*)
element_wise = a * b
print(f"Element-wise multiplication (a * b):\n{element_wise}")

# 2. Matrix Multiplication (@ or tf.matmul)
# Equivalent to PyTorch @ or torch.matmul
matrix_product = a @ b # or tf.matmul(a, b)
print(f"Matrix Multiplication (a @ b):\n{matrix_product}")


# =====================================================================
# CONCEPT 4: Autograd via GradientTape (Crucial Framework Difference!)
# =====================================================================
print("\n--------------------------------------------------")
print("CONCEPT 4: Autograd via GradientTape")
print("--------------------------------------------------")
# Crucial Difference: PyTorch tracks gradients implicitly.
# TensorFlow requires you to explicitly record operations inside a `tf.GradientTape` block!

# Define weights and inputs
w = tf.Variable([3.0])
x = tf.constant([2.0])
b = tf.Variable([1.0])

# We open a GradientTape to record the forward pass operations
with tf.GradientTape() as tape:
    # Forward pass: y = w * x + b
    y = w * x + b # y = 3 * 2 + 1 = 7

# Calculate gradients of y with respect to w and b
# Equivalent to PyTorch's y.backward()
dy_dw, dy_db = tape.gradient(y, [w, b])

print(f"Forward output y: {y.numpy()}")
print(f"Gradient of w (dy/dw): {dy_dw.numpy()} (Should be equal to x = 2.0)")
print(f"Gradient of b (dy/db): {dy_db.numpy()} (Should be 1.0)")


# =====================================================================
# CONCEPT 5: The tf.keras Model Blueprint (Custom Layers)
# =====================================================================
print("\n--------------------------------------------------")
print("CONCEPT 5: The tf.keras Model Blueprint")
print("--------------------------------------------------")
# In TensorFlow, custom layers inherit from `tf.keras.layers.Layer`, 
# and models inherit from `tf.keras.Model` (equivalent to PyTorch's nn.Module).
# Instead of `forward()`, we implement `call()`.

class MyTFLinearLayer(object): 
    # Under standard Keras: class MyTFLinearLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features):
        # We define weights inside init or build()
        super().__init__()
        # Initialize weights randomly
        self.w = tf.Variable(tf.random.normal((in_features, out_features)) * 0.01, name="w")
        self.b = tf.Variable(tf.zeros((1, out_features)), name="b")

    def call(self, x):
        # The 'call' method is equivalent to PyTorch's 'forward'
        return x @ self.w + self.b

# Instantiate custom layer
my_layer = MyTFLinearLayer(in_features=3, out_features=2)
print("Initialized custom TensorFlow Linear Layer.")
print(f"Weight shape: {my_layer.w.shape} | Bias shape: {my_layer.b.shape}")


# =====================================================================
# CONCEPT 6: The Custom Training Loop Recipe
# =====================================================================
print("\n--------------------------------------------------")
print("CONCEPT 6: The Custom Training Loop Recipe")
print("--------------------------------------------------")
# This is the TensorFlow recipe for custom training loops (similar to PyTorch's 5 steps).

# 1. Create dummy dataset
inputs = tf.random.normal((10, 3))
targets = tf.random.normal((10, 2))

# 2. Instantiate model and optimizer
model = MyTFLinearLayer(in_features=3, out_features=2)
optimizer = tf.optimizers.SGD(learning_rate=0.1) if hasattr(tf, 'optimizers') else None

print("TensorFlow Custom Training Loop Blueprint:")
print("1. Open tf.GradientTape() context.")
print("2. Run forward pass: predictions = model(inputs).")
print("3. Calculate loss: loss = loss_fn(predictions, targets).")
print("4. Get gradients: grads = tape.gradient(loss, model.trainable_variables).")
print("5. Apply updates: optimizer.apply_gradients(zip(grads, model.trainable_variables)).")
print("\n==================================================")
print("🎉 CONGRATULATIONS! YOU NOW UNDERSTAND TENSORFLOW BASICS!")
print("==================================================")
