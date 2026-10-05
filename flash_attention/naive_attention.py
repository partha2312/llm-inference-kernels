import torch
import math

class Transformer:
    def __init__(self, d, H):
        self.d = d
        self.H = H
        self.ffn_dim = 4 * d
        # trainable matrices in transformer
        self.W_q = torch.randn((d, d), dtype=torch.float32)
        self.W_k = torch.randn((d, d), dtype=torch.float32)
        self.W_v = torch.randn((d, d), dtype=torch.float32)
        self.W_o = torch.randn((d, d), dtype=torch.float32)
        # trainable matrices in FFN
        self.W_explode = torch.randn((d, self.ffn_dim), dtype=torch.float32)
        self.W_compress = torch.randn((self.ffn_dim, d), dtype=torch.float32)
        # trainable vectors in layer norms
        self.norm_gamma_attention = torch.ones(d)
        self.norm_beta_attention = torch.zeros(d)
        self.norm_gamma_ffn = torch.ones(d)
        self.norm_beta_ffn = torch.zeros(d)

    def layer_norm(self, q):
        mean = torch.mean(q, axis=q.dim()-1, keepdim=True)
        var = torch.var(q, axis=q.dim()-1, keepdim=True) + 1e-8
        return (q - mean) / torch.sqrt(var)

    def causal_mask(self, q):
        _, _, r, c = q.shape
        i, j = torch.meshgrid(torch.arange(r), torch.arange(c), indexing='ij')
        return torch.where(j <= i, q, -float("inf"))
    
    def relu(self, q):
        return torch.clamp(q, min=0.0)

    def softmax(self, q):
        values, _ = torch.max(q, axis=q.dim()-1, keepdim=True)
        q_shifted = q - values
        q_exp = torch.exp(q_shifted)
        return q_exp / torch.sum(q_exp, axis=q.dim()-1, keepdim=True)

    def attention(self, Q, K, V, d_k):
        q_kt = Q @ K.transpose(2, 3)
        q_kt_scaled_masked = self.causal_mask(q_kt / math.sqrt(d_k))
        return self.softmax(q_kt_scaled_masked) @ V
    
    def multi_headed_attention(self, X):
        B, N = X.shape[0], X.shape[1]
        Q, K, V = X @ self.W_q, X @ self.W_k, X @ self.W_v # B x N x d
        d_k = self.d // self.H
        '''
        reshape: B x N x d -> B x N x H x d_k
        transpose: B x N x H x d_k -> B x H x N x d_k
        '''
        Q = Q.reshape(B, N, self.H, d_k).transpose(1, 2)
        K = K.reshape(B, N, self.H, d_k).transpose(1, 2)
        V = V.reshape(B, N, self.H, d_k).transpose(1, 2)
        r = self.attention(Q, K, V, d_k)
        '''
        transpose: B x H x N x d_k -> B x N x H x d_k
        reshape: B x N x H x d_k -> B x N x d
        '''
        return r.transpose(1, 2).reshape(B, N, -1)
    
    def feed_forward(self, transformer_op):
        return self.relu(transformer_op @ self.W_explode) @ self.W_compress
    
    def transform(self, X):
        attention = self.multi_headed_attention(X)
        attention_normed = self.layer_norm((attention @ self.W_o) + X)
        ffn_input = self.norm_gamma_attention * attention_normed + self.norm_beta_attention
        ffn_normed = self.layer_norm(self.feed_forward(ffn_input) + ffn_input)
        return self.norm_gamma_ffn * ffn_normed + self.norm_beta_ffn



B, N, d, H = 1, 1024, 128, 1
transformer = Transformer(d, H)
# input embedding, can't be rand for size N
X = torch.randn((B, N, d), dtype=torch.float32) # B x N x d
transformer_op = transformer.transform(X)
print(transformer_op.shape)