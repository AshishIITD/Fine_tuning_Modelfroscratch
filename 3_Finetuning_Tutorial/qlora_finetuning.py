# -*- coding: utf-8 -*-
"""
Module 3e: PEFT Type 3 - QLoRA (Quantized Low-Rank Adaptation) from Scratch

QLoRA is an advanced fine-tuning technique that reduces memory usage even further 
than LoRA by combining two concepts:
1. Quantization: Base model weights are quantized (e.g., to 8-bit or 4-bit integers), 
   dramatically reducing the storage and memory footprint of the base model.
2. LoRA Adapters: A tiny set of high-precision (float32) trainable adapter weights 
   are injected into the model.

During training:
*   The quantized base weights are completely frozen (requires_grad = False).
*   During the forward pass, the quantized weights are de-quantized to float32 on-the-fly,
    multiplied with the input, and then added to the high-precision LoRA adapter path.
*   Only the float32 LoRA adapter weights receive gradients and are updated.

In this script:
1. We load a base model pre-trained on general facts.
2. We write a custom QLoRALinear PyTorch wrapper class from scratch, implementing 
   symmetric 8-bit integer quantization.
3. We freeze the entire model and inject QLoRALinear adapters into our attention blocks.
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
# SECTION 1: QLoRALinear Layer Wrapper from Scratch (8-bit Quantization)
# =====================================================================
class QLoRALinear(nn.Module):
    """
    A wrapper layer that quantizes the base linear weights into 8-bit integers,
    de-quantizes them on-the-fly during the forward pass, and adds trainable 
    float32 LoRA adapters.
    """
    def __init__(self, original_linear, r=4, alpha=8):
        super(QLoRALinear, self).__init__()
        
        # Get dimensions
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.scaling = alpha / r
        
        # 1. Symmetric 8-bit Quantization of base weights
        W = original_linear.weight.data
        
        # Calculate scale factor: map max absolute weight value to 127 (range of signed 8-bit integer: [-127, 127])
        max_val = torch.max(torch.abs(W))
        self.scale = max_val / 127.0
        
        # Quantize the weights: round(W / scale) and clamp to 8-bit integer boundaries
        # We cast the tensor to torch.int8 to demonstrate actual low-precision storage!
        W_quant = torch.clamp(torch.round(W / self.scale), -127, 127).to(torch.int8)
        
        # Register W_quant as a buffer so it is saved with the model but not updated by gradients
        self.register_buffer("W_quant", W_quant)
        
        # Save bias as a frozen float32 buffer if it exists
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data)
        else:
            self.bias = None
            
        # 2. Define the low-rank trainable parameters in high precision (float32)
        # lora_A: initialized with random normal
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        # lora_B: initialized with zeros so the adapter starts inactive
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

    def forward(self, x):
        # 1. De-quantize the base weight on-the-fly to float32
        # W_dequant = W_quant * scale
        W_dequant = self.W_quant.to(x.dtype) * self.scale
        
        # 2. Base forward pass: x @ W_dequant.T
        base_out = torch.matmul(x, W_dequant.t())
        if self.bias is not None:
            base_out = base_out + self.bias
            
        # 3. LoRA adapter forward pass: (x @ lora_A) @ lora_B * scaling
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scaling
        
        # Sum both paths
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
    
    # 4. Inject QLoRA adapters
    print("\nStarting PEFT - QLORA FINE-TUNING on Pirate Q&A...")
    print("Freezing the entire model and injecting QLoRALinear layers (8-bit quantized)...")
    
    # Freeze the ENTIRE model first
    for param in model.parameters():
        param.requires_grad = False
        
    # Traverse the blocks and swap the Attention QKV projection with a QLoRALinear wrapper
    for block in model.blocks:
        block.attn.qkv_projection = QLoRALinear(block.attn.qkv_projection, r=4, alpha=8)
        
    # Let's count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (QLoRA): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Check the data type of the base weight to prove it is int8!
    for block in model.blocks:
        print(f"Base attention weight storage type: {block.attn.qkv_projection.W_quant.dtype}")
        break
        
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
    print("\n--- EVALUATION AFTER QLORA FINE-TUNING ---")
    model.eval()
    for q in ["Q: who are you? ", "Q: where is gold? "]:
        idx = torch.tensor([encode(q)], dtype=torch.long)
        res = model.generate(idx, max_new_tokens=22, temperature=0.1)
        print(f"Prompt: {q}")
        print(f"Model:  {decode(res[0].tolist()).split(q)[-1]}\n")
