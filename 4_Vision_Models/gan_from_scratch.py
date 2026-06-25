# -*- coding: utf-8 -*-
"""
Module 4b: Generative Adversarial Network (GAN) from Scratch

In this file, we build a complete Generative Adversarial Network (GAN) from scratch.
A GAN consists of two models playing a minimax game:
1. The Generator: Learns to create realistic images starting from random noise.
2. The Discriminator: Learns to distinguish between real images and generated fake images.

To make this visual and terminal-friendly, we train the Generator to draw a 
cross shape 'X' and print the output as ASCII art at the start vs. the end of training.

Formulas:
    Minimax Game Objective:
        min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# SECTION 1: Generator and Discriminator Architectures
# =====================================================================
class Generator(nn.Module):
    """
    Takes a low-dimensional latent noise vector and expands it into a 28x28 image.
    """
    def __init__(self, latent_dim=10, img_size=784):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, img_size),
            nn.Tanh() # Scales pixel outputs between [-1.0, 1.0]
        )

    def forward(self, z):
        # z shape: [batch, latent_dim]
        # returns shape: [batch, 784]
        return self.net(z)


class Discriminator(nn.Module):
    """
    Takes a 28x28 image (flattened to 784) and outputs a probability (0 to 1) 
    of whether the image is real or fake.
    """
    def __init__(self, img_size=784):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(img_size, 128),
            nn.LeakyReLU(0.2, inplace=True), # Standard for GAN discriminators
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid() # Outputs probability between [0.0, 1.0]
        )

    def forward(self, img):
        # img shape: [batch, 784]
        # returns shape: [batch, 1]
        return self.net(img)


# =====================================================================
# SECTION 2: Dataset and Visualizer Helpers
# =====================================================================
def get_real_cross_data(batch_size=64):
    """
    Generates a batch of perfect crosses 'X' scaled between [-1.0, 1.0] to match Tanh.
    """
    images = []
    for _ in range(batch_size):
        img = np.zeros((28, 28), dtype=np.float32)
        # Draw cross lines with a slight offset
        offset_y = np.random.randint(-1, 2)
        offset_x = np.random.randint(-1, 2)
        cy, cx = 14 + offset_y, 14 + offset_x
        
        for i in range(-6, 7):
            img[cy + i, cx + i] = 1.0
            img[cy + i, cx - i] = 1.0
            
        # Rescale from [0, 1] to [-1, 1] (required for Tanh generator output)
        img = (img * 2.0) - 1.0
        images.append(img.flatten())
        
    return torch.tensor(images, dtype=torch.float32)


def render_ascii_art(flat_img):
    """
    Converts a flat [-1, 1] pixel tensor into a beautiful terminal ASCII drawing.
    """
    img = flat_img.view(28, 28).numpy()
    chars = " .:-=+*#%@"
    art = []
    for r in range(28):
        row_chars = []
        for c in range(28):
            # Scale pixel from [-1, 1] to [0, 9] to index the characters
            val = int((img[r, c] + 1.0) * 4.5)
            val = max(0, min(9, val))
            row_chars.append(chars[val])
        art.append("".join(row_chars))
    return "\n".join(art)


# =====================================================================
# SECTION 3: Training Loop
# =====================================================================
if __name__ == "__main__":
    latent_dim = 10
    batch_size = 64
    epochs = 200
    
    print("==================================================")
    print("🎭 INITIALIZING GENERATIVE ADVERSARIAL NETWORK (GAN)")
    print("==================================================")
    
    # Instantiate models
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    
    # 1. Show starting random generation (Epoch 0)
    print("\n[GEN 1] Generator output BEFORE training (Epoch 0 - Pure Noise):")
    with torch.no_grad():
        test_noise = torch.randn(1, latent_dim)
        fake_img = generator(test_noise)[0]
        print(render_ascii_art(fake_img))
    print("--------------------------------------------------")
    
    print("\nTraining GAN to draw a cross shape 'X'...")
    for epoch in range(1, epochs + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Real images
        real_imgs = get_real_cross_data(batch_size)
        real_labels = torch.ones(batch_size, 1) # Real = 1
        
        # Fake images
        noise = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(noise)
        fake_labels = torch.zeros(batch_size, 1) # Fake = 0
        
        # Discriminator forward & loss
        d_optimizer.zero_grad()
        
        real_preds = discriminator(real_imgs)
        d_loss_real = criterion(real_preds, real_labels)
        
        # We detach fake_imgs so gradients do not propagate back to generator during D training
        fake_preds = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(fake_preds, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # ---------------------
        #  Train Generator
        # ---------------------
        g_optimizer.zero_grad()
        
        # Generator wants the Discriminator to believe the fake images are real (1.0)
        output = discriminator(fake_imgs)
        g_loss = criterion(output, real_labels)
        
        g_loss.backward()
        g_optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            
    # 2. Show final trained generation (Epoch 200)
    print("\n==================================================")
    print("[GEN 2] Generator output AFTER training (Epoch 200):")
    print("==================================================")
    with torch.no_grad():
        # Generate with the same latent noise to see the dramatic change!
        final_fake_img = generator(test_noise)[0]
        print(render_ascii_art(final_fake_img))
    print("==================================================")
    print("\n[KEY GAN TAKEAWAY]")
    print("1. Notice how the initial output was completely random static.")
    print("2. After dueling the Discriminator, the Generator learned to bundle high-pixel values")
    # In ASCII art, characters like '@' and '#' represent high pixel values, making the 'X' shape clearly visible!
    print("   into diagonal lines, drawing a clear cross 'X' shape starting from pure random numbers!")
