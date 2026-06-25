# -*- coding: utf-8 -*-
"""
Module 3a: Full Fine-Tuning

In full fine-tuning, we update 100% of the model's weights during the training phase.
While this yields high accuracy, it requires a lot of memory and computation because 
we must calculate gradients and maintain optimizer states for every single parameter.

In this script:
1. We load a base model pre-trained on general facts.
2. We test it to show that it does not know the pirate persona.
3. We perform full fine-tuning where ALL parameters are updated.
4. We verify that the model successfully learns to answer like a pirate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Dynamically add the Module 2 directory to allow clean imports of our scratch Transformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../2_Transformer_From_Scratch')))
from transformer_from_scratch import MiniGPT, get_batch

# Set seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# Datasets Setup (Shared Vocabulary)
# =====================================================================
general_corpus = """The sky is blue and bright.
Cats say meow in the night.
Dogs bark loud and play.
Birds fly high and away.
Fish swim deep in the sea.
Rain falls down on the tree."""

pirate_qa_dataset = """Q: who are you? A: i am a pirate matey!
Q: where is gold? A: buried on the island!
Q: what is name? A: captain blackbeard!
Q: how are you? A: sailing the high seas!
Q: who are you? A: i am a pirate matey!
Q: where is gold? A: buried on the island!"""

full_text = general_corpus + "\n" + pirate_qa_dataset
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_to_ix[c] for c in s]
decode = lambda l: "".join([ix_to_char[i] for i in l])

pretrain_data = torch.tensor(encode(general_corpus), dtype=torch.long)
finetune_data = torch.tensor(encode(pirate_qa_dataset), dtype=torch.long)

# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    context_len = 32
    embed_dim = 64
    num_heads = 4
    num_layers = 3
    
    # 1. Initialize Model
    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, context_len)
    
    # 2. Pre-train the model on General Facts (Simulating a base LLM)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    print("Pre-training base model on general facts...")
    model.train()
    for epoch in range(1, 801):
        xb, yb = get_batch(pretrain_data, batch_size=16, context_len=context_len)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # 3. Test Base Model
    print("\n--- BASE MODEL EVALUATION (BEFORE FINE-TUNING) ---")
    prompt = "Q: who are you? "
    prompt_idx = torch.tensor([encode(prompt)], dtype=torch.long)
    base_response = model.generate(prompt_idx, max_new_tokens=25, temperature=0.1)
    print(f"Prompt: '{prompt}'")
    print(f"Response: '{decode(base_response[0].tolist())}'")
    print("--------------------------------------------------")
    
    # 4. Full Fine-Tuning
    print("\nStarting FULL FINE-TUNING on Pirate Q&A...")
    print("Updating ALL weights (no parameters are frozen)...")
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Trainable Parameters: {total_params:,} (100% of the model)")
    
    # Fine-tuning loop
    ft_optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(1, 601):
        xb, yb = get_batch(finetune_data, batch_size=8, context_len=context_len)
        logits, loss = model(xb, yb)
        ft_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ft_optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Fine-tune Epoch {epoch:3d}/600 | Loss: {loss.item():.4f}")
            
    # 5. Evaluate Fine-tuned Model
    print("\n--- EVALUATION AFTER FULL FINE-TUNING ---")
    model.eval()
    for q in ["Q: who are you? ", "Q: where is gold? "]:
        idx = torch.tensor([encode(q)], dtype=torch.long)
        res = model.generate(idx, max_new_tokens=22, temperature=0.1)
        print(f"Prompt: {q}")
        print(f"Model:  {decode(res[0].tolist()).split(q)[-1]}\n")
