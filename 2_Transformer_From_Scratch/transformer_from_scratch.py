# -*- coding: utf-8 -*-
"""
Module 2: Building a Small Transformer LLM (Decoder-Only GPT-Style) from Scratch

In this module, we step away from recurrent architectures (RNNs/LSTMs) and build a
modern, parallelized Decoder-Only Transformer (MiniGPT) from scratch in PyTorch.

We will write every key block ourselves:
1. Multi-Head Causal Self-Attention
2. Position-wise Feed-Forward Network
3. Pre-LayerNorm Transformer Block
4. Token & Positional Embeddings
5. Autoregressive Text Generation

=========================================
OUR TASK: Learning a Nursery Rhyme
=========================================
We will train our MiniGPT on a classic nursery rhyme so it learns grammar, punctuation, 
and spelling, and can generate new lines when given a starting prompt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# Set seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# SECTION 1: Causal Multi-Head Self-Attention
# =====================================================================
class CausalSelfAttention(nn.Module):
    """
    Computes Multi-Head Attention with a causal mask to prevent looking into the future.
    """
    def __init__(self, embed_dim, num_heads, context_len):
        super(CausalSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # We project the input into Queries (Q), Keys (K), and Values (V) using a single linear layer
        # shape: [embed_dim, 3 * embed_dim]
        self.qkv_projection = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        
        # Output projection back to the embedding dimension
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        
        # Causal mask: Lower triangular matrix of ones.
        # Any element above the diagonal is set to 0. During forward, 0s will be masked to -inf.
        # We register this as a buffer so it is saved with the model but not updated by gradients.
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embed_dim] (denoted as B, T, C)
        Returns:
            out (Tensor): Attention output of shape [batch_size, seq_len, embed_dim]
        """
        B, T, C = x.size()
        
        # 1. Project to Q, K, V
        # qkv shape: [B, T, 3 * C]
        qkv = self.qkv_projection(x)
        
        # Split into individual Query, Key, and Value tensors
        # Each has shape: [B, T, C]
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        # 2. Reshape for Multi-Head Attention
        # We split the channel dimension C into [num_heads, head_dim]
        # Then transpose from [B, T, nh, hs] -> [B, nh, T, hs] so we can perform matrix multiplication on heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # shape: [B, nh, T, hs]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # shape: [B, nh, T, hs]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # shape: [B, nh, T, hs]
        
        # 3. Compute scaled dot-product attention scores
        # scores = (Q @ K^T) / sqrt(head_dim)
        # shape of Q: [B, nh, T, hs]
        # shape of K^T: [B, nh, hs, T] (transposed last two dimensions)
        # scores shape: [B, nh, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Apply Causal Masking
        # We fill positions where the bias matrix is 0 with a very small number (-1e9)
        # This makes the softmax probability for future tokens exactly 0.
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # 5. Softmax along the last dimension to get attention weights (probabilities)
        # attn_weights shape: [B, nh, T, T]
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. Multiply attention weights by Values
        # out shape: [B, nh, T, hs]
        out = torch.matmul(attn_weights, v)
        
        # 7. Reassemble (concatenate) all heads back into a single vector
        # Transpose back: [B, nh, T, hs] -> [B, T, nh, hs]
        # contiguous() ensures memory layout is correct before reshape
        # out shape: [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Final linear projection
        return self.out_projection(out)


# =====================================================================
# SECTION 2: Position-wise Feed-Forward Network (FFN)
# =====================================================================
class FeedForward(nn.Module):
    """
    A simple multilayer perceptron applied to each token position individually.
    Formula:
        FFN(x) = GELU(x * W1 + b1) * W2 + b2
    """
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        # GPT architectures typically expand the hidden dimension by 4x inside the FFN
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU() # Smooth approximation of ReLU
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        # Input shape: [B, T, C]
        # Output shape: [B, T, C]
        return self.fc2(self.gelu(self.fc1(x)))


# =====================================================================
# SECTION 3: The Transformer Block (Pre-LN)
# =====================================================================
class TransformerBlock(nn.Module):
    """
    A single Transformer layer combining Multi-Head Attention and a Feed-Forward Network,
    wrapped with residual connections and Layer Normalization (Pre-LN style).
    """
    def __init__(self, embed_dim, num_heads, context_len):
        super(TransformerBlock, self).__init__()
        # Normalization layer before self-attention
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-Head Attention
        self.attn = CausalSelfAttention(embed_dim, num_heads, context_len)
        # Normalization layer before Feed-Forward
        self.ln2 = nn.LayerNorm(embed_dim)
        # Feed-Forward Network
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        # x shape: [B, T, C]
        
        # 1. Self-Attention with residual connection (Pre-LN style)
        x = x + self.attn(self.ln1(x))
        
        # 2. Feed-Forward with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x


# =====================================================================
# SECTION 4: The Full Decoder-Only LLM (MiniGPT)
# =====================================================================
class MiniGPT(nn.Module):
    """
    A complete Decoder-Only language model that predicts the next token.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, context_len):
        super(MiniGPT, self).__init__()
        self.context_len = context_len
        
        # Token embedding: maps each character index to a dense vector
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding: learns a vector representation for each position in the context window
        self.position_embedding = nn.Embedding(context_len, embed_dim)
        
        # A sequence of stacked Transformer Blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, context_len) for _ in range(num_layers)
        ])
        
        # Final LayerNorm before logits classification
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language Model classification head: maps hidden state to vocabulary logits
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        """
        Args:
            idx (Tensor): Input token indices of shape [B, T]
            targets (Tensor, optional): Ground truth next token indices of shape [B, T]
        Returns:
            logits (Tensor): Predicted token scores of shape [B, T, vocab_size]
            loss (Tensor, optional): Cross entropy loss if targets are provided
        """
        B, T = idx.size()
        assert T <= self.context_len, f"Cannot process sequence of length {T}, context window is only {self.context_len}"
        
        # 1. Generate position indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0) # shape: [1, T]
        
        # 2. Add Token and Position embeddings together
        # token_embeddings: [B, T, C]
        # pos_embeddings: [1, T, C] (broadcasted to B)
        x = self.token_embedding(idx) + self.position_embedding(positions) # shape: [B, T, C]
        
        # 3. Pass through the stack of Transformer blocks
        x = self.blocks(x) # shape: [B, T, C]
        
        # 4. Apply final normalization
        x = self.ln_f(x) # shape: [B, T, C]
        
        # 5. Project to vocabulary logits
        logits = self.lm_head(x) # shape: [B, T, vocab_size]
        
        loss = None
        if targets is not None:
            # Flatten logits and targets for cross entropy computation
            # logits: [B * T, vocab_size]
            # targets: [B * T]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generates new text autoregressively starting from a seed prompt indices (idx).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # If the current sequence is longer than the context window, crop it
            idx_cond = idx if idx.size(1) <= self.context_len else idx[:, -self.context_len:]
            
            # Forward pass to get logits for the last token in the sequence
            logits, _ = self(idx_cond) # shape: [B, T, vocab_size]
            
            # Focus only on the last time step logits and apply temperature scaling
            # logits shape becomes: [B, vocab_size]
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probability distribution
            probs = F.softmax(logits, dim=-1) # shape: [B, vocab_size]
            
            # Sample from the distribution (adds variety/creativity to text)
            next_idx = torch.multinomial(probs, num_samples=1) # shape: [B, 1]
            
            # Append sampled token to the running sequence
            idx = torch.cat((idx, next_idx), dim=1) # shape: [B, T + 1]
            
        return idx


# =====================================================================
# SECTION 5: Character Tokenizer and Data Setup
# =====================================================================

# A small classic nursery rhyme to train our model on
nursery_rhyme = """Jack and Jill went up the hill,
To fetch a pail of water.
Jack fell down and broke his crown,
And Jill came tumbling after."""

print("--- TRAINING TEXT ---")
print(nursery_rhyme)
print("---------------------\n")

# Build Vocabulary
chars = sorted(list(set(nursery_rhyme)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Encoder: convert string to list of integers
encode = lambda s: [char_to_ix[c] for c in s]
# Decoder: convert list of integers to string
decode = lambda l: "".join([ix_to_char[i] for i in l])

# Convert entire rhyme to a tensor of tokens
data = torch.tensor(encode(nursery_rhyme), dtype=torch.long)
print(f"Total Characters in Dataset: {len(data)}")
print(f"Unique Characters (Vocab Size): {vocab_size}\n")


# =====================================================================
# SECTION 6: Creating Training Batches
# =====================================================================
def get_batch(data, batch_size, context_len):
    """
    Generates a small batch of inputs (X) and targets (Y).
    X is the context sequence, and Y is the same sequence shifted by 1 token (the next token).
    """
    # Randomly select starting indices for the batch
    # We must ensure we leave room to extract 'context_len' tokens + 1 target token
    ix = torch.randint(len(data) - context_len, (batch_size,))
    
    # Extract context sequences
    x = torch.stack([data[i:i+context_len] for i in ix])
    # Extract target sequences (shifted by 1)
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    
    return x, y


# =====================================================================
# SECTION 7: Let's Train and Generate!
# =====================================================================
if __name__ == "__main__":
    # Hyperparameters for our MiniGPT model
    batch_size = 16
    context_len = 16    # Size of the context window (how many tokens the model can look at)
    embed_dim = 64      # Token embedding dimension
    num_heads = 4       # Multi-head attention heads (64 / 4 = 16 dimensions per head)
    num_layers = 3      # Number of Transformer blocks stacked
    
    epochs = 1200
    learning_rate = 3e-3
    
    # Instantiate the model
    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, context_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("--------------------------------------------------")
    print(f"INITIALIZING MINIGPT")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("--------------------------------------------------\n")
    
    # Let's generate before training to see what a completely random model outputs
    print("--- SAMPLE GENERATION (BEFORE TRAINING) ---")
    start_prompt = "Jack "
    start_idx = torch.tensor([encode(start_prompt)], dtype=torch.long)
    generated_idx = model.generate(start_idx, max_new_tokens=50)
    print(decode(generated_idx[0].tolist()))
    print("-------------------------------------------\n")
    
    # Training Loop
    print("--- TRAINING MINIGPT ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 1. Get a batch of training data
        xb, yb = get_batch(data, batch_size, context_len)
        
        # 2. Forward pass: compute logits and loss
        logits, loss = model(xb, yb)
        
        # 3. Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Print loss and sample progress
        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f}")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds!\n")
    
    # Let's generate after training to see what it learned!
    print("--- SAMPLE GENERATION (AFTER TRAINING) ---")
    # We will run generation twice with different temperatures:
    # 1. Temperature = 0.2 (Very focused and deterministic)
    # 2. Temperature = 1.0 (More creative and diverse)
    
    print("\n[Gen 1] Temperature 0.2 (Focused & Greedy-like):")
    gen_focused = model.generate(start_idx, max_new_tokens=80, temperature=0.2)
    print(decode(gen_focused[0].tolist()))
    
    print("\n[Gen 2] Temperature 1.0 (Creative & Varied):")
    gen_creative = model.generate(start_idx, max_new_tokens=80, temperature=1.0)
    print(decode(gen_creative[0].tolist()))
    print("-------------------------------------------\n")
    
    print("[TAKEAWAYS FOR THE BEGINNER]")
    print("1. Look at the generation before training vs. after. The random model outputs gibberish, but the trained model understands spelling, spacing, capitalization, and rhyme structure!")
    print("2. Multi-Head Attention allows the model to process all character relationships simultaneously, unlike sequential RNNs.")
    print("3. Notice the role of Temperature: lower values make the text highly predictable (greedy), while higher values add creativity but might cause spelling errors.")
    
    # Save the trained model weights
    import os
    model_path = os.path.join(os.path.dirname(__file__), "minigpt_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model weights successfully saved to: {model_path}")
