import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e # number of embeddings
        self.e_dim = e_dim # dimension of embeddings
        self.beta = beta # commitment cost

        # initialize codebook
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.uniform_(self.embedding.weight.data, -1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # convert z (B, C, H, W) to (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        # flatten z to (B*H*W, C) for comparison with codebook
        z_flat = z.view(-1, self.e_dim)

        # Count the euclidean distance between z and the embeddings
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        # Find the index of the closest embeddings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        # Convert the indices to one-hot encodings
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        # scatter_(dim, index, src)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantize the z
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Compute the loss for the quantization
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Compute the quantized version of z. Detach separates the gradient computation
        # z_q is updated, but the gradient is not propagated to z
        z_q = z + (z_q - z).detach()

        # Compute the perplexity, perplexity is a measure of how well the model is learning
        e_mean = torch.mean(min_encodings, dim=0)   
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Reshape z_q back to (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Return the quantized z, loss, perplexity, min_encodings, and min_encoding_indices
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
    
    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e, device=indices.device)
        min_encodings.scatter_(1, indices[:, None], 1)

        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            
        return z_q
