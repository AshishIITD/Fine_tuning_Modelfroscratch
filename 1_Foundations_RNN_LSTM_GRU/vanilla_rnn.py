# -*- coding: utf-8 -*-
"""
Module 1a: Vanilla Recurrent Neural Network (RNN) from Scratch

In this file, we build a Vanilla RNN from scratch in PyTorch using raw tensor math,
explaining the core mathematics and training it to predict characters.

Formula for a Vanilla RNN Cell:
    h_t = tanh( W_hh * h_{t-1} + W_xh * x_t + b_h )
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# SECTION 1: Vanilla RNN Cell
# =====================================================================
class VanillaRNNCell(nn.Module):
    """
    A single Vanilla RNN cell.
    """
    def __init__(self, input_dim, hidden_dim):
        super(VanillaRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Define weights and biases as parameters
        # W_xh: Weight matrix for input x_t
        # W_hh: Weight matrix for previous hidden state h_{t-1}
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t (Tensor): Input at time t of shape [batch_size, input_dim]
            h_prev (Tensor): Previous hidden state of shape [batch_size, hidden_dim]
        Returns:
            h_next (Tensor): Next hidden state of shape [batch_size, hidden_dim]
        """
        # Linear projection of input and previous state, summed with bias, activated by tanh
        # h_next = tanh( x_t @ W_xh + h_prev @ W_hh + b_h )
        h_next = torch.tanh(torch.matmul(x_t, self.W_xh) + torch.matmul(h_prev, self.W_hh) + self.b_h)
        return h_next


# =====================================================================
# SECTION 2: Character-Level Sequence Predictor
# =====================================================================
class CharacterRNNModel(nn.Module):
    """
    Model wrapper that processes a sequence of tokens step-by-step
    using our custom VanillaRNNCell.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharacterRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Map character index to dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Custom cell
        self.rnn_cell = VanillaRNNCell(embed_dim, hidden_dim)
        # Project hidden state back to vocab size to predict next character
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h_init=None):
        """
        Args:
            x (Tensor): Input indices of shape [batch_size, seq_len]
            h_init (Tensor, optional): Initial hidden state
        """
        batch_size, seq_len = x.size()
        embedded = self.embedding(x) # shape: [batch_size, seq_len, embed_dim]
        
        if h_init is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h = h_init
            
        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :] # Input vector at time t: [batch_size, embed_dim]
            h = self.rnn_cell(x_t, h) # Step cell
            out_t = self.fc(h) # Predict next character: [batch_size, vocab_size]
            outputs.append(out_t)
            
        return torch.stack(outputs, dim=1) # Stack to [batch_size, seq_len, vocab_size]


# =====================================================================
# SECTION 3: Dataset and Training
# =====================================================================
if __name__ == "__main__":
    # Target text
    text = "learning deep learning is fun!"
    print(f"Target Text: '{text}'")
    
    # Vocabulary mappings
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    char_to_ix = {char: i for i, char in enumerate(vocab)}
    ix_to_char = {i: char for i, char in enumerate(vocab)}
    
    text_indices = [char_to_ix[c] for c in text]
    x_data = torch.tensor(text_indices[:-1]).unsqueeze(0) # input
    y_data = torch.tensor(text_indices[1:]).unsqueeze(0)  # target (shifted by 1)
    
    # Instantiate model
    embed_dim = 16
    hidden_dim = 32
    model = CharacterRNNModel(vocab_size, embed_dim, hidden_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    print("--- TRAINING VANILLA RNN ---")
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(x_data)
        loss = criterion(predictions.view(-1, vocab_size), y_data.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            # Generate sample
            model.eval()
            with torch.no_grad():
                curr_char_idx = char_to_ix['l']
                result = ['l']
                h = torch.zeros(1, hidden_dim)
                for _ in range(len(text) - 1):
                    x_t_idx = torch.tensor([[curr_char_idx]])
                    embedded_x = model.embedding(x_t_idx)[:, 0, :]
                    h = model.rnn_cell(embedded_x, h)
                    logits = model.fc(h)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    result.append(ix_to_char[pred_idx])
                    curr_char_idx = pred_idx
                generated_str = "".join(result)
            print(f"Epoch {epoch:3d}/300 | Loss: {loss.item():.4f} | Generated: '{generated_str}'")
