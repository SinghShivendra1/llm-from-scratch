import torch
import torch.nn as nn
class SingleHeadAttention(nn.module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        # mbedding_dim: Dimension of the input token representation.
        # attention_dim: Dimension of Query, Key and Value.

        super().__init__()
        torch.manual_seed(0)
        self.get_queries = nn.linear(embedding_dim, attention_dim, bias = False)
        self.get_keys = nn.linear(embedding_dim, attention_dim, bias = False)
        self.get_values = nn.linear(embedding_dim, attention_dim, bias = False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.get_keys(embedded) #B, T, A
        q = self.get_queries(embedded)
        v = self.get_values(embedded)

        # matrix multiplication
        scores = q @ torch.transpose(k, 1, 2)
        # we also need value of attention dim, as we know, we actually want to take the square root and divide by it
        # So, we actually get the attention dim from the variety of places
        # easiest way to get : it is just k (B, T, A).
        B, T, A = k.shape
        scores = scores / (A**0.5)

        # pre-mask
        pre_mask = torch.tril(torch.ones(T, T))
        mask = pre_mask == 0;
        scores = scores.masked_fill(mask, float('-inf'))
        # below dim=2, means we want to apply softmax row wise(across columns)
        scores = nn.functional.softmax(scores, dim = 2) # B, T, T
        transformed = scores @ v
        return torch.round(transformed, decimals = 4)