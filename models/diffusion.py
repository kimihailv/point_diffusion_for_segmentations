import torch
import torch.nn as nn
import math
from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self,
                 extractor, total_steps, time_embedding_dim=384):
        super().__init__()
        self.extractor = extractor
        self.total_steps = total_steps
        betas = torch.linspace(
            0.0001, 0.008, total_steps, dtype=torch.float32
        )

        self.time_embedding_dim = time_embedding_dim

        '''alphas_cum = torch.arange(total_steps) / self.total_steps + 0.008
        alphas_cum /= 1 + 0.008
        alphas_cum = torch.cos(alphas_cum * math.pi/2).pow(2)
        alphas_cum = alphas_cum / alphas_cum[0]
        alphas_cum_prev = torch.cat([torch.ones(1), alphas_cum[:-1]], dim=0)
        alphas = alphas_cum / alphas_cum_prev
        alphas = torch.clamp_min(alphas, 0.001)
        betas = 1 - alphas
        variance = betas * (1 - alphas_cum_prev) / (1 - alphas_cum + 1e-8)

        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('alphas', alphas)
        self.register_buffer('variance', variance)'''

        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)

        sigma = torch.zeros_like(betas)

        for i in range(1, total_steps):
            sigma[i - 1] = ((1 - alphas_cum[i - 1]) / (1 - alphas_cum[i])) * betas[i]

        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cum', alphas_cum)
        self.register_buffer('sigma', sigma.sqrt())

    def forward(self, x):
        batch_size = x.size(0)
        # timesteps = torch.randperm(self.total_steps, device=x.device)[:batch_size]
        '''timesteps = torch.randint(
            low=0, high=self.total_steps, size=(batch_size // 2 + 1,)
        ).to(x.device)'''
        timesteps = torch.randint(low=0, high=self.total_steps, size=(batch_size,), device=x.device)
        # timesteps = torch.cat([timesteps, self.total_steps - timesteps - 1], dim=0)[:batch_size]
        time_emb = self.get_time_embeddings(self.time_embedding_dim, timesteps)
        xt, e = self.sample_xt(x, timesteps)
        et = self.extractor(x, xt, time_emb, return_features=False)

        return e, et

    def sample_xt(self, x0, timesteps):
        a = torch.index_select(self.alphas_cum, 0, timesteps).view(-1, 1, 1)
        e = torch.randn_like(x0)
        return x0 * a.sqrt() + e * (1 - a).sqrt(), e

    @torch.no_grad()
    def sample(self, x_current=None, x0=None, batch_size=8,
               track_trajectory=True):
        batch_size = batch_size if x_current is None else x_current.size(0)

        if x_current is None:
            x_current = torch.randn(batch_size, 3, 2048, device=self.alphas.device)

        if track_trajectory:
            trajectory = [x_current.detach().cpu()]

        latent = None

        for timestep in tqdm(range(self.total_steps - 1, -1, -1)):
            if track_trajectory:
                trajectory.append(x_current.detach().cpu())

            timesteps = torch.IntTensor([timestep]).to(self.alphas.device).expand(batch_size)
            time_emb = self.get_time_embeddings(self.time_embedding_dim, timesteps)
            et, ctx, (latent, _, _) = self.extractor(x0, x_current, time_emb, return_features=False, z=latent)

            c = (1 - self.alphas[timestep]) / (1 - self.alphas_cum[timestep])**0.5
            z = 0 if timestep == 0 else torch.randn_like(x_current, device=self.alphas.device)
            x_current = (x_current - c * et) / self.alphas[timestep]**0.5 + self.sigma[timestep] * z

        if track_trajectory:
            return x_current, trajectory

        return x_current

    @torch.no_grad()
    def get_features(self, x, timesteps):
        z = None

        features_list = {}
        coords_list = {}
        for timestep in tqdm(timesteps):
            t = torch.IntTensor([timestep]).to(self.alphas.device).expand(x.size(0))
            xt, _ = self.sample_xt(x, t)
            time_emb = self.get_time_embeddings(self.time_embedding_dim, t)
            (features, coords), (et, ctx, (z, _, _)) = self.extractor(x, xt, time_emb, return_features=True, z=z)
            features_list[timestep] = features
            coords_list[timestep] = coords

        return features_list, coords_list

    def get_time_embeddings(self, dim, timesteps):
        '''half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)'''
        beta = torch.index_select(self.betas, 0, timesteps).unsqueeze(1)
        embedding = torch.cat([beta, beta.sin(), beta.cos()], dim=1)
        return embedding
