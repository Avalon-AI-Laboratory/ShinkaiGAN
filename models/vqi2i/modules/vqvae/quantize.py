import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)

        d = torch.sum(z_flat**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flat, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
    
    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e, device=indices.device)
        min_encodings.scatter_(1, indices[:, None], 1)

        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            
        return z_q
