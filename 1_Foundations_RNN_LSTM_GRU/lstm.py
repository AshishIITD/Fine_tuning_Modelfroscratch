# -*- coding: utf-8 -*-
"""
Module 1b: Long Short-Term Memory (LSTM) from Scratch

In this file, we build an LSTM cell from scratch in PyTorch using raw tensor math,
explaining the gating mechanisms (Forget, Input, Output) and cell state highway,
and training it to predict characters.

Formulas for a single LSTM cell:
    f_t = sigmoid( W_xf * x_t + W_hf * h_{t-1} + b_f )  (Forget Gate)
    i_t = sigmoid( W_xi * x_t + W_hi * h_{t-1} + b_i )  (Input Gate)
    g_t = tanh( W_xg * x_t + W_hg * h_{t-1} + b_g )     (Candidate Cell State)
    c_t = f_t * c_{t-1} + i_t * g_t                    (New Cell State highway)
    o_t = sigmoid( W_xo * x_t + W_ho * h_{t-1} + b_o )  (Output Gate)
    h_t = o_t * tanh(c_t)                               (New Hidden State)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# SECTION 1: LSTM Cell
# =====================================================================
class LSTMCell(nn.Module):
    """
    A single LSTM cell.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # For clean code, we combine the weights of all 4 gates (Forget, Input, Candidate, Output)
        # into a single parameter matrix of size [input_dim, 4 * hidden_dim].
        self.W_x = nn.Parameter(torch.randn(input_dim, 4 * hidden_dim) * 0.01)
        self.W_h = nn.Parameter(torch.randn(hidden_dim, 4 * hidden_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1, 4 * hidden_dim))

    def forward(self, x_t, states):
        """
        Args:
            x_t (Tensor): Input at time t of shape [batch_size, input_dim]
            states (Tuple): (h_prev, c_prev) where:
                            h_prev is shape [batch_size, hidden_dim]
                            c_prev is shape [batch_size, hidden_dim]
        Returns:
            h_next (Tensor): Next hidden state [batch_size, hidden_dim]
            c_next (Tensor): Next cell state [batch_size, hidden_dim]
        """
        h_prev, c_prev = states

        # Project input and previous hidden state into the 4x hidden space
        gates_proj = torch.matmul(x_t, self.W_x) + torch.matmul(h_prev, self.W_h) + self.b

        # Split the projection into 4 equal chunks for the 4 gates
        f_gate_proj, i_gate_proj, g_gate_proj, o_gate_proj = torch.chunk(gates_proj, 4, dim=1)

        # Apply activation functions to get gate values
        f_t = torch.sigmoid(f_gate_proj)  # Forget Gate (keep/drop past memory)
        i_t = torch.sigmoid(i_gate_proj)  # Input Gate (what new info to write)
        g_t = torch.tanh(g_gate_proj)     # Candidate state (new memories)
        o_t = torch.sigmoid(o_gate_proj)  # Output Gate (what to export to hidden state)

        # Update Cell State (highway update!)
        c_next = f_t * c_prev + i_t * g_t

        # Update Hidden State (filtered cell state)
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


# =====================================================================
# SECTION 2: Character-Level Sequence Predictor
# =====================================================================
class CharacterLSTMModel(nn.Module):
    """
    Model wrapper using our custom LSTMCell.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharacterLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = LSTMCell(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, states=None):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        
        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h, c = states
            
        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h, c = self.lstm_cell(x_t, (h, c))
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
    model = CharacterLSTMModel(vocab_size, embed_dim, hidden_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    print("--- TRAINING LSTM ---")
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
                c = torch.zeros(1, hidden_dim)
                for _ in range(len(text) - 1):
                    x_t_idx = torch.tensor([[curr_char_idx]])
                    embedded_x = model.embedding(x_t_idx)[:, 0, :]
                    h, c = model.lstm_cell(embedded_x, (h, c))
                    logits = model.fc(h)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    result.append(ix_to_char[pred_idx])
                    curr_char_idx = pred_idx
                generated_str = "".join(result)
            print(f"Epoch {epoch:3d}/300 | Loss: {loss.item():.4f} | Generated: '{generated_str}'")
