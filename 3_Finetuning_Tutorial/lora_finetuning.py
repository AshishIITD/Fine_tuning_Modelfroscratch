# -*- coding: utf-8 -*-
"""
Module 3c: PEFT Type 2 - LoRA (Low-Rank Adaptation) from Scratch

LoRA (Low-Rank Adaptation) is one of the most popular and powerful parameter-efficient 
fine-tuning (PEFT) methods used in the industry.

Core Concept:
Instead of updating the full weight matrix W (shape [d, k]) of a linear layer, we freeze W
and inject two small, low-rank matrices A (shape [d, r]) and B (shape [r, k]) where the rank r << d.
The forward pass changes from:
    h = x @ W
to:
    h = x @ W + (x @ A @ B) * (alpha / r)

By choosing a small rank (e.g., r = 4), we update less than 1% of the model's parameters 
while achieving performance that matches or exceeds full fine-tuning!

In this script:
1. We load a base model pre-trained on general facts.
2. We write a custom LoRALinear PyTorch wrapper class from scratch.
3. We freeze the entire model and inject LoRALinear adapters into our attention blocks.
4. We train ONLY the LoRA adapter weights on our pirate dataset.
5. We verify that the model successfully learns to answer like a pirate.
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
# SECTION 1: LoRALinear Layer Wrapper from Scratch
# =====================================================================
class LoRALinear(nn.Module):
    """
    A wrapper layer that freezes the original linear weights and adds 
    trainable low-rank decomposition matrices A and B.
    """
    def __init__(self, original_linear, r=4, alpha=8):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear
        
        # 1. Freeze the original pre-trained weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.scaling = alpha / r
        
        # 2. Define the low-rank trainable parameters
        # lora_A is initialized with random normal values to break symmetry
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        # lora_B is initialized with ZEROS. 
        # This guarantees that at step 0 of training, A * B = 0, meaning the model's
        # behavior is EXACTLY identical to the base pre-trained model!
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

    def forward(self, x):
        # Standard base projection: shape [B, T, out_features]
        base_out = self.original_linear(x)
        
        # LoRA delta projection: (x @ lora_A) @ lora_B * scaling
        # x shape: [B, T, in_features]
        # lora_A shape: [in_features, r]
        # lora_B shape: [r, out_features]
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scaling
        
        # Return the sum of base and adapter paths
        return base_out + lora_out


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
    
    # 4. Inject LoRA adapters
    print("\nStarting PEFT - LORA FINE-TUNING on Pirate Q&A...")
    print("Freezing the entire model and injecting LoRALinear layers...")
    
    # Freeze the ENTIRE model first
    for param in model.parameters():
        param.requires_grad = False
        
    # Now, traverse our Transformer blocks and swap the Attention QKV projection with a LoRALinear wrapper!
    # In each block, block.attn.qkv_projection is a standard nn.Linear.
    for block in model.blocks:
        block.attn.qkv_projection = LoRALinear(block.attn.qkv_projection, r=4, alpha=8)
        
    # Let's count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (LoRA): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Fine-tuning loop (only train the active LoRA parameters!)
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
    print("\n--- EVALUATION AFTER LORA FINE-TUNING ---")
    model.eval()
    for q in ["Q: who are you? ", "Q: where is gold? "]:
        idx = torch.tensor([encode(q)], dtype=torch.long)
        res = model.generate(idx, max_new_tokens=22, temperature=0.1)
        print(f"Prompt: {q}")
        print(f"Model:  {decode(res[0].tolist()).split(q)[-1]}\n")
