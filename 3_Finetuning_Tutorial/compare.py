# -*- coding: utf-8 -*-
"""
Module 3d: Fine-Tuning Comparative Benchmark (Including QLoRA)

This script pre-trains a single base model on general facts, creates four 
exact deep copies of it, and runs:
1. Full Fine-Tuning (All weights updated, float32)
2. Freeze Tuning (Only final head updated, float32)
3. LoRA Tuning (Only low-rank adapters updated, float32)
4. QLoRA Tuning (Base weights quantized to 8-bit integers, only adapters updated)

It benchmarks the training times, the number of active parameters, 
and prints a final comparative performance table.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import sys
import os

# Dynamically add the Module 2 directory to allow clean imports of our scratch Transformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../2_Transformer_From_Scratch')))
from transformer_from_scratch import MiniGPT, get_batch

# Set seed for reproducibility
torch.manual_seed(42)

# Import our custom adapters
from lora_finetuning import LoRALinear
from qlora_finetuning import QLoRALinear

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
# Benchmark Runner
# =====================================================================
if __name__ == "__main__":
    context_len = 32
    embed_dim = 64
    num_heads = 4
    num_layers = 3
    
    # 1. Initialize and load the pre-trained shared base model
    print("==================================================")
    print("STAGE 1: LOADING PRE-TRAINED WEIGHTS FROM MODULE 2")
    print("==================================================")
    shared_base_model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, context_len)
    
    saved_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../2_Transformer_From_Scratch/minigpt_model.pt'))
    
    if os.path.exists(saved_model_path):
        print(f"Found saved model checkpoint at: {saved_model_path}")
        checkpoint = torch.load(saved_model_path)
        
        # Partial Weight Loading:
        # We filter out embeddings and lm_head because the vocabulary sizes differ
        # (Module 2 has 27 characters, while Module 3 has more due to Q&A punctuation).
        # We load 100% of the learned self-attention, FFN, and LayerNorm blocks!
        backbone_state_dict = {}
        for k, v in checkpoint.items():
            if "token_embedding" not in k and "position_embedding" not in k and "lm_head" not in k:
                if k in shared_base_model.state_dict() and shared_base_model.state_dict()[k].shape == v.shape:
                    backbone_state_dict[k] = v
                    
        shared_base_model.load_state_dict(backbone_state_dict, strict=False)
        print("Successfully loaded pre-trained Transformer backbone from your Module 2 checkpoint!")
    else:
        print(f"⚠️ Saved model checkpoint not found at {saved_model_path}.")
        print("Falling back to training a base model on-the-fly...")
        optimizer = optim.AdamW(shared_base_model.parameters(), lr=2e-3)
        shared_base_model.train()
        for epoch in range(1, 600):
            xb, yb = get_batch(pretrain_data, batch_size=16, context_len=context_len)
            logits, loss = shared_base_model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print("Fallback pre-training complete.")
        
    # Test base response (should output generic text before fine-tuning)
    prompt = "Q: who are you? "
    prompt_idx = torch.tensor([encode(prompt)], dtype=torch.long)
    base_response = decode(shared_base_model.generate(prompt_idx, max_new_tokens=22, temperature=0.1)[0].tolist())
    print(f"\nBase model response to prompt '{prompt}':")
    print(f"-> '{base_response}'\n")

    # 2. Setup the four comparative runs
    print("==================================================")
    print("STAGE 2: RUNNING COMPARATIVE FINE-TUNING BENCHMARKS")
    print("==================================================")
    
    # Define common fine-tuning function
    def run_finetuning(model, trainable_params_filter, lr=5e-3, epochs=600):
        optimizer = optim.AdamW(trainable_params_filter, lr=lr)
        
        start_time = time.time()
        model.train()
        for epoch in range(1, epochs + 1):
            xb, yb = get_batch(finetune_data, batch_size=8, context_len=context_len)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        training_time = time.time() - start_time
        
        # Evaluate completions
        model.eval()
        completions = []
        for q in ["Q: who are you? ", "Q: where is gold? "]:
            idx = torch.tensor([encode(q)], dtype=torch.long)
            res = model.generate(idx, max_new_tokens=20, temperature=0.1)
            ans = decode(res[0].tolist()).split(q)[-1].strip().replace("\n", " ")
            completions.append(f"'{ans}'")
            
        return loss.item(), training_time, completions

    # --- RUN 1: FULL FINE-TUNING ---
    print("Running Full Fine-Tuning...")
    model_full = copy.deepcopy(shared_base_model)
    total_params = sum(p.numel() for p in model_full.parameters())
    trainable_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    loss_full, time_full, comps_full = run_finetuning(
        model_full, 
        model_full.parameters(), 
        lr=1e-3
    )

    # --- RUN 2: FREEZE TUNING (HEAD ONLY) ---
    print("Running Freeze Tuning...")
    model_freeze = copy.deepcopy(shared_base_model)
    # Freeze backbone
    model_freeze.token_embedding.weight.requires_grad = False
    model_freeze.position_embedding.weight.requires_grad = False
    for param in model_freeze.blocks.parameters():
        param.requires_grad = False
    model_freeze.lm_head.weight.requires_grad = True
    if model_freeze.lm_head.bias is not None:
        model_freeze.lm_head.bias.requires_grad = True
        
    trainable_freeze = sum(p.numel() for p in model_freeze.parameters() if p.requires_grad)
    loss_freeze, time_freeze, comps_freeze = run_finetuning(
        model_freeze, 
        filter(lambda p: p.requires_grad, model_freeze.parameters()), 
        lr=5e-3
    )

    # --- RUN 3: LORA TUNING (FROM SCRATCH) ---
    print("Running LoRA Fine-Tuning...")
    model_lora = copy.deepcopy(shared_base_model)
    # Freeze entire model
    for param in model_lora.parameters():
        param.requires_grad = False
    # Inject LoRA adapters
    for block in model_lora.blocks:
        block.attn.qkv_projection = LoRALinear(block.attn.qkv_projection, r=4, alpha=8)
        
    trainable_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    loss_lora, time_lora, comps_lora = run_finetuning(
        model_lora, 
        filter(lambda p: p.requires_grad, model_lora.parameters()), 
        lr=5e-3
    )

    # --- RUN 4: QLORA TUNING (8-BIT QUANTIZED FROM SCRATCH) ---
    print("Running QLoRA Fine-Tuning...")
    model_qlora = copy.deepcopy(shared_base_model)
    # Freeze entire model
    for param in model_qlora.parameters():
        param.requires_grad = False
    # Inject 8-bit QLoRA adapters
    for block in model_qlora.blocks:
        block.attn.qkv_projection = QLoRALinear(block.attn.qkv_projection, r=4, alpha=8)
        
    trainable_qlora = sum(p.numel() for p in model_qlora.parameters() if p.requires_grad)
    loss_qlora, time_qlora, comps_qlora = run_finetuning(
        model_qlora, 
        filter(lambda p: p.requires_grad, model_qlora.parameters()), 
        lr=5e-3
    )

    # =====================================================================
    # Print Benchmarks
    # =====================================================================
    print("\n==================================================")
    print("FINAL FINE-TUNING BENCHMARK TABLE")
    print("==================================================")
    print(f"Fine-Tuning Method | Trainable Params | % Trainable | Train Time | Final Loss | Sample Responses (Q1 & Q2)")
    print(f"---------------------------------------------------------------------------------------------------------")
    
    pct_full = (trainable_full / total_params) * 100
    pct_freeze = (trainable_freeze / total_params) * 100
    
    total_params_lora = sum(p.numel() for p in model_lora.parameters())
    pct_lora = (trainable_lora / total_params_lora) * 100
    
    total_params_qlora = sum(p.numel() for p in model_qlora.parameters())
    pct_qlora = (trainable_qlora / total_params_qlora) * 100
    
    print(f"Full Fine-Tuning   | {trainable_full:<16,d} | {pct_full:<11.1f}% | {time_full:<9.2f}s | {loss_full:<10.4f} | {', '.join(comps_full)}")
    print(f"Freeze Tuning      | {trainable_freeze:<16,d} | {pct_freeze:<11.1f}% | {time_freeze:<9.2f}s | {loss_freeze:<10.4f} | {', '.join(comps_freeze)}")
    print(f"LoRA (From Scratch)| {trainable_lora:<16,d} | {pct_lora:<11.1f}% | {time_lora:<9.2f}s | {loss_lora:<10.4f} | {', '.join(comps_lora)}")
    print(f"QLoRA (8-bit Quant)| {trainable_qlora:<16,d} | {pct_qlora:<11.1f}% | {time_qlora:<9.2f}s | {loss_qlora:<10.4f} | {', '.join(comps_qlora)}")
    print("==================================================")
    print("\n[KEY TAKEAWAY FOR THE BEGINNER]")
    print("1. **Full Fine-Tuning** trains 100% of the weights, which is slow and memory-intensive.")
    print("2. **Freeze Tuning** is fast and only trains a tiny layer, but has less capacity to learn complex styles.")
    print("3. **LoRA** trains a very small fraction of weights (only 1.9% in our case) by adding low-rank matrices.")
    print("4. **QLoRA** quantizes base weights into 8-bit integers (reducing weight memory footprint by 75%),")
    # Note: float32 = 4 bytes, int8 = 1 byte. 1 / 4 = 0.25, which is a 75% reduction!
    print("   while using float32 adapters to adapt the model. It matches LoRA's accuracy but uses a fraction of the RAM!")
