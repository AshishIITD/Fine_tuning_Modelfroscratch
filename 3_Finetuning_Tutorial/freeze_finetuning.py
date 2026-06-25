# -*- coding: utf-8 -*-
"""
Module 3b: PEFT Type 1 - Freeze Tuning (Head Tuning)

Freeze tuning is the simplest parameter-efficient fine-tuning technique.
We freeze the entire pre-trained backbone of the network (representing 95%+ of the weights)
and only train the final classification head (the language modeling output layer).

Because the backbone weights are frozen, we do not need to calculate gradients for them 
or store their optimizer states, saving significant memory and compute!

In this script:
1. We load a base model pre-trained on general facts.
2. We freeze the embedding layer and all Transformer Blocks.
3. We train ONLY the final classification layer (lm_head).
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
    
    # 4. Freeze Backbone
    print("\nStarting PEFT - FREEZE TUNING on Pirate Q&A...")
    print("Freezing embedding layers and all Transformer blocks...")
    
    # Freeze token embeddings, position embeddings, and blocks
    model.token_embedding.weight.requires_grad = False
    model.position_embedding.weight.requires_grad = False
    for param in model.blocks.parameters():
        param.requires_grad = False
        
    # Only the final classification head (lm_head) will be trained!
    # Let's verify that lm_head is still trainable
    model.lm_head.weight.requires_grad = True
    if model.lm_head.bias is not None:
        model.lm_head.bias.requires_grad = True
        
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (PEFT): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Fine-tuning loop (only pass trainable parameters to optimizer!)
    ft_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=5e-3
    )
    
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
    print("\n--- EVALUATION AFTER FREEZE TUNING ---")
    model.eval()
    for q in ["Q: who are you? ", "Q: where is gold? "]:
        idx = torch.tensor([encode(q)], dtype=torch.long)
        res = model.generate(idx, max_new_tokens=22, temperature=0.1)
        print(f"Prompt: {q}")
        print(f"Model:  {decode(res[0].tolist()).split(q)[-1]}\n")
