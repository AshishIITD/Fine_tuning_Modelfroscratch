# -*- coding: utf-8 -*-
"""
Module 1d: Sequential Model Comparative Benchmark

This script imports our custom from-scratch Vanilla RNN, LSTM, and GRU models
from their respective independent files, trains them on a shared task,
and displays a comparative performance table.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

# Dynamically add the current directory to Python's search path to allow clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the custom models from our separate files
from vanilla_rnn import CharacterRNNModel
from lstm import CharacterLSTMModel
from gru import CharacterGRUModel

# Set random seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# Dataset Setup
# =====================================================================
text = "learning deep learning is fun!"
print("==================================================")
print(f"TARGET TEXT: '{text}'")
print("==================================================\n")

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
char_to_ix = {char: i for i, char in enumerate(vocab)}
ix_to_char = {i: char for i, char in enumerate(vocab)}

text_indices = [char_to_ix[c] for c in text]
x_data = torch.tensor(text_indices[:-1]).unsqueeze(0) # shape: [1, seq_len-1]
y_data = torch.tensor(text_indices[1:]).unsqueeze(0)  # shape: [1, seq_len-1]

# =====================================================================
# Benchmark Training Function
# =====================================================================
def train_and_evaluate(model, model_type, epochs=350):
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(x_data)
        loss = criterion(predictions.view(-1, vocab_size), y_data.view(-1))
        
        loss.backward()
        optimizer.step()
        
    total_time = time.time() - start_time
    
    # Evaluate and generate sample
    model.eval()
    with torch.no_grad():
        curr_char_idx = char_to_ix['l']
        result = ['l']
        
        # Initialize states based on model type
        h = torch.zeros(1, model.hidden_dim)
        if model_type == "LSTM":
            c = torch.zeros(1, model.hidden_dim)
            
        for _ in range(len(text) - 1):
            x_t_idx = torch.tensor([[curr_char_idx]])
            embedded_x = model.embedding(x_t_idx)[:, 0, :]
            
            # Run cell based on model type
            if model_type == "RNN":
                h = model.rnn_cell(embedded_x, h)
            elif model_type == "LSTM":
                h, c = model.lstm_cell(embedded_x, (h, c))
            elif model_type == "GRU":
                h = model.gru_cell(embedded_x, h)
                
            logits = model.fc(h)
            pred_idx = torch.argmax(logits, dim=1).item()
            result.append(ix_to_char[pred_idx])
            curr_char_idx = pred_idx
            
        generated_str = "".join(result)
        
    return loss.item(), total_time, generated_str

# =====================================================================
# Run Benchmark
# =====================================================================
if __name__ == "__main__":
    embed_dim = 16
    hidden_dim = 32
    epochs = 400
    
    print("Training Vanilla RNN...")
    rnn_model = CharacterRNNModel(vocab_size, embed_dim, hidden_dim)
    rnn_loss, rnn_time, rnn_gen = train_and_evaluate(rnn_model, "RNN", epochs)
    
    print("Training LSTM...")
    lstm_model = CharacterLSTMModel(vocab_size, embed_dim, hidden_dim)
    lstm_loss, lstm_time, lstm_gen = train_and_evaluate(lstm_model, "LSTM", epochs)
    
    print("Training GRU...")
    gru_model = CharacterGRUModel(vocab_size, embed_dim, hidden_dim)
    gru_loss, gru_time, gru_gen = train_and_evaluate(gru_model, "GRU", epochs)
    
    print("\n==================================================")
    print("BENCHMARK COMPARISON TABLE")
    print("==================================================")
    print(f"Model       | Final Loss | Training Time | Perfect Reconstruction?")
    
    def check_perfect(generated):
        return "Yes!" if generated == text else "No"

    print(f"Vanilla RNN | {rnn_loss:.4f}     | {rnn_time:.2f}s        | {check_perfect(rnn_gen)} (Got: '{rnn_gen}')")
    print(f"LSTM        | {lstm_loss:.4f}     | {lstm_time:.2f}s        | {check_perfect(lstm_gen)} (Got: '{lstm_gen}')")
    print(f"GRU         | {gru_loss:.4f}     | {gru_time:.2f}s        | {check_perfect(gru_gen)} (Got: '{gru_gen}')")
    print("==================================================")
