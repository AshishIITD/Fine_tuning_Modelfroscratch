# -*- coding: utf-8 -*-
"""
Module 1c: Gated Recurrent Unit (GRU) from Scratch

In this file, we build a GRU cell from scratch in PyTorch using raw tensor math,
explaining the two gates (Update, Reset) and how they simplify LSTMs,
and training it to predict characters.

Formulas for a single GRU cell:
    z_t = sigmoid( W_xz * x_t + W_hz * h_{t-1} + b_z )  (Update Gate: how much past memory to retain)
    r_t = sigmoid( W_xr * x_t + W_hr * h_{t-1} + b_r )  (Reset Gate: how much past memory to ignore)
    n_t = tanh( W_xn * x_t + r_t * (W_hn * h_{t-1} + b_hn) + b_xn ) (Candidate state)
    h_t = (1 - z_t) * h_{t-1} + z_t * n_t               (New Hidden State)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# SECTION 1: GRU Cell
# =====================================================================
class GRUCell(nn.Module):
    """
    A single GRU cell.
    """
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Combine weights for Update (z) and Reset (r) gates into matrices of size 2*hidden_dim
        self.W_x_gates = nn.Parameter(torch.randn(input_dim, 2 * hidden_dim) * 0.01)
        self.W_h_gates = nn.Parameter(torch.randn(hidden_dim, 2 * hidden_dim) * 0.01)
        self.b_gates = nn.Parameter(torch.zeros(1, 2 * hidden_dim))

        # Weights for Candidate Hidden State (n)
        self.W_xn = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.W_hn = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_xn = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b_hn = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t (Tensor): Input tensor of shape [batch_size, input_dim]
            h_prev (Tensor): Previous hidden state of shape [batch_size, hidden_dim]
        Returns:
            h_next (Tensor): Next hidden state of shape [batch_size, hidden_dim]
        """
        # 1. Compute update (z) and reset (r) gates
        gates_proj = torch.matmul(x_t, self.W_x_gates) + torch.matmul(h_prev, self.W_h_gates) + self.b_gates
        z_gate_proj, r_gate_proj = torch.chunk(gates_proj, 2, dim=1)
        
        z_t = torch.sigmoid(z_gate_proj)  # Update gate
        r_t = torch.sigmoid(r_gate_proj)  # Reset gate

        # 2. Compute candidate hidden state (n_t)
        # Note how the reset gate r_t acts directly on the previous hidden state projection
        n_t = torch.tanh(
            torch.matmul(x_t, self.W_xn) + self.b_xn + 
            r_t * (torch.matmul(h_prev, self.W_hn) + self.b_hn)
        )

        # 3. Interpolate between previous state and candidate state
        h_next = (1 - z_t) * h_prev + z_t * n_t
        return h_next


# =====================================================================
# SECTION 2: Character-Level Sequence Predictor
# =====================================================================
class CharacterGRUModel(nn.Module):
    """
    Model wrapper using our custom GRUCell.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharacterGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru_cell = GRUCell(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h_init=None):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        
        if h_init is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h = h_init
            
        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h = self.gru_cell(x_t, h)
            out_t = self.fc(h)
            outputs.append(out_t)
            
        return torch.stack(outputs, dim=1)


# =====================================================================
# SECTION 3: Dataset and Training
# =====================================================================
if __name__ == "__main__":
    text = "learning deep learning is fun!"
    print(f"Target Text: '{text}'")
    
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    char_to_ix = {char: i for i, char in enumerate(vocab)}
    ix_to_char = {i: char for i, char in enumerate(vocab)}
    
    text_indices = [char_to_ix[c] for c in text]
    x_data = torch.tensor(text_indices[:-1]).unsqueeze(0)
    y_data = torch.tensor(text_indices[1:]).unsqueeze(0)
    
    embed_dim = 16
    hidden_dim = 32
    model = CharacterGRUModel(vocab_size, embed_dim, hidden_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    print("--- TRAINING GRU ---")
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
                    h = model.gru_cell(embedded_x, h)
                    logits = model.fc(h)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    result.append(ix_to_char[pred_idx])
                    curr_char_idx = pred_idx
                generated_str = "".join(result)
            print(f"Epoch {epoch:3d}/300 | Loss: {loss.item():.4f} | Generated: '{generated_str}'")
