import torch
import torch.nn as nn

# Input embeddings: 3 tokens, embedding dimension 2
X = torch.tensor([
    [1., 0.],
    [0., 1.],
    [1., 1.]
])

# Define weight matrices
Wq = torch.tensor([
    [1., 0.],
    [0., 1.]
])

Wk = torch.tensor([
    [1., 1.],
    [0., 1.]
])

Wv = torch.tensor([
    [1., 0.],
    [1., 1.]
])

# Calculate Q, K, V
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Calculate attention scores
scores = Q @ K.T

# Scale
A = K.shape[-1]
scores = scores / (A ** 0.5)

# Causal mask
T = X.shape[0]
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)

scores = scores.masked_fill(mask, float('-inf'))

# Softmax
attention_weights = torch.softmax(scores, dim=-1)

# Final output
output = attention_weights @ V

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("Attention scores:\n", scores)
print("Attention weights:\n", attention_weights)
print("Final output:\n", output)
