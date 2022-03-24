import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMLP(nn.Module):
    def __init__(self, in_channels, hid_channels, time_emb_dim):
        super().__init__()
        self.mlp = []

        for i, channels in enumerate(hid_channels, start=1):
            self.mlp.append(nn.Conv1d(in_channels, channels, 1))
            self.mlp.append(nn.BatchNorm1d(channels))

            # self.layers.append(nn.LeakyReLU())
            if i < len(hid_channels):
                self.mlp.append(nn.LeakyReLU(negative_slope=0.02))

            in_channels = channels

        self.mlp = nn.Sequential(*self.mlp)

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, 512),
            nn.SiLU(),
            nn.Linear(512, in_channels),
        )

    def forward(self, x, time_emb):
        time_emb = self.time_embed(time_emb).unsqueeze(2).expand(-1, -1, x.size(2))
        return self.mlp(x) + time_emb


class AttentionMix(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()

        self.q = nn.Conv1d(in_channels, in_channels * num_heads, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels * num_heads, 1, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels * num_heads, 1, bias=False)
        self.out = nn.Conv1d(in_channels * num_heads, in_channels, 1, bias=False)
        self.num_heads = num_heads
        self.norm_const = in_channels ** .5

    def forward(self, x, y, mask=None):
        # x: b x c x k
        # y: b x c x k x n

        bs = x.size(0)
        x_pts = x.size(2)
        total_pts = y.size(3)

        q = self.q(x).view(bs, self.num_heads, -1, x_pts)  # b x h x c x k
        k = self.k(y).view(bs, self.num_heads, -1, x_pts, total_pts)  # b x h x c x k x n
        v = self.v(y).view(bs, self.num_heads, -1, x_pts, total_pts)  # b x h x c x k x n

        w = torch.einsum('bhck, bhckn -> bhkn', q, k) / self.norm_const  # b x h x k x n
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(1), -1e9)

        w = F.softmax(w, dim=3)
        out = torch.einsum('bhkn, bhckn -> bhck', w, v).contiguous().view(bs, -1, x_pts)  # b x h x c x k

        return self.out(out) + x


class Attention1d(nn.Module):
    def __init__(self, q_channels, k_channels, out_channels, num_heads=2):
        super().__init__()
        self.q = nn.Conv1d(q_channels, out_channels * num_heads, 1, bias=False)
        self.v = nn.Conv1d(k_channels, out_channels * num_heads, 1, bias=False)
        self.k = nn.Conv1d(k_channels, out_channels * num_heads, 1, bias=False)
        self.out = nn.Conv1d(out_channels * num_heads, out_channels, 1, bias=False)
        self.num_heads = num_heads
        self.norm_const = out_channels**.5

    def forward(self, x, y):
        bs = x.size(0)
        x_pts = x.size(2)
        y_pts = y.size(2)
        # q: bs x n_heads x Nx x hid_size
        q = self.q(x).view(bs, self.num_heads, -1, x_pts).transpose(-1, -2)
        # k: bs x n_heads x hid_size x Ny
        k = self.k(y).view(bs, self.num_heads, -1, y_pts)
        # v: bs x n_heads x Ny x hid_size
        v = self.v(y).view(bs, self.num_heads, -1, y_pts).transpose(-1, -2)
        # bs x n_heads x Nx x Ny
        logits = torch.einsum('bnik, bnkj->bnij', q, k) / self.norm_const
        attention_probs = F.softmax(logits, dim=-1)
        # out: bs x n_heads * hid_size x Nx
        out = torch.einsum('bnik, bnkj->bnji', attention_probs, v).contiguous().view(bs, -1, x_pts)
        return self.out(out)


class PointNetMSG(nn.Module):
    def __init__(self, n_centroids, max_n_points, radius, in_channels, hid_channels, time_emb_dim):
        super().__init__()
        self.n_centroids = n_centroids
        self.max_n_points = max_n_points
        self.radius = radius
        self.mlps = nn.ModuleList()
        self.attention = nn.ModuleList()

        for channels in hid_channels:
            self.attention.append(AttentionMix(in_channels))
            self.mlps.append(TemporalMLP(in_channels, channels, time_emb_dim))

    def forward(self, xyz, time_emb, point_features=None):
        """
        :param xyz: point cloud coordinates, b x 3 x n
        :param point_features: pointwise features, b x c x n
        :return: sample of xyz, new features
        """
        features_list = []
        support = xyz if point_features is None else torch.cat([xyz, point_features], dim=1)

        centroids_idx = self.sample(support)  # b x n_centroids
        centroids = torch.gather(support, 2, centroids_idx.unsqueeze(1).expand(-1, support.size(1), -1))
        new_xyz = torch.gather(xyz, 2, centroids_idx.unsqueeze(1).expand(-1, xyz.size(1), -1))
        ex_support = support.unsqueeze(2).expand(-1, -1, self.n_centroids, -1)  # b x (c+3) x n_centroids x n
        for radius, k, mlp, attn in zip(self.radius, self.max_n_points, self.mlps, self.attention):
            group_idx, mask = self.group(support, centroids, radius, k)
            # b x c x n_centroids x n_points
            group = torch.gather(ex_support, 3, group_idx.unsqueeze(1).expand(-1, support.size(1), -1, -1))
            group -= centroids.unsqueeze(3)

            features = attn(centroids, group, mask)
            features = mlp(features, time_emb)
            features_list.append(features)

        features = torch.cat(features_list, dim=1)

        return new_xyz, features

    def sample(self, x):
        device = x.device
        B, C, N = x.shape
        centroids = torch.zeros(B, self.n_centroids, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(self.n_centroids):
            centroids[:, i] = farthest
            centroid = x[batch_indices, :, farthest].view(B, C, 1)
            dist = torch.sum((x - centroid) ** 2, 1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def group(self, x, centroids, radius, k):
        # x: b x c x n
        # centroids: b x c x n_centroids

        batch_size, n_points = x.size(0), x.size(2)
        n_centroids = centroids.size(2)

        dists = (
                x.pow(2).sum(dim=1, keepdim=True) -
                2 * torch.bmm(centroids.transpose(2, 1), x)
                + centroids.pow(2).sum(dim=1).unsqueeze(2)
        )  # b x m x n

        idx = torch.arange(n_points, device=x.device).view(1, 1, n_points).expand(batch_size, n_centroids, -1).clone()
        idx[dists > radius**2] = n_points
        idx = torch.topk(idx, k, dim=2, largest=False)[0]  # deterministic neighbourhood size restriction
        first_point_idx = idx[:, :, 0:1].expand(batch_size, n_centroids, k)
        mask = idx == n_points
        idx[mask] = first_point_idx[mask]

        # b x n_centroids x k
        return idx, mask


class FeaturePropagation(nn.Module):
    def __init__(self, x_in_channels, y_in_channels, hid_channels, time_emb_dim):
        super().__init__()
        self.layers = []

        self.attention = Attention1d(x_in_channels + 3, y_in_channels + 3, y_in_channels)
        in_channels = x_in_channels + y_in_channels
        self.mlp = TemporalMLP(in_channels, hid_channels, time_emb_dim)

    def forward(self, x, y, x_features, y_features, time_emb):
        """ =

        if x_features is not None:
            interpolated = torch.cat((interpolated.sum(dim=3), x_features), dim=1)
        else:
            interpolated = interpolated.sum(dim=3)
        return self.layers(interpolated) """

        x = torch.cat((x, x_features), dim=1) if x_features is not None else x
        y = torch.cat((y, y_features), dim=1)

        interpolated = self.attention(x, y)
        if x_features is not None:
            interpolated = torch.cat((interpolated, x_features), dim=1)

        return self.mlp(interpolated, time_emb)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv1d(in_channels, in_channels, 1)
        self.k = nn.Conv1d(in_channels, in_channels, 1)
        self.v = nn.Conv1d(in_channels, in_channels, 1)
        self.out = nn.Conv1d(in_channels, in_channels, 1)
        self.norm = 1 / in_channels**0.5

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        w = torch.bmm(q.transpose(2, 1), k) * self.norm
        w = F.softmax(w, dim=2)  # b x n x n
        v = self.out(torch.bmm(v, w.transpose(2, 1)))

        return x + v


class PointNetPP(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()

        self.down1 = PointNetMSG(n_centroids=1024,
                                 max_n_points=[16, 32],
                                 radius=[0.05, 0.1],
                                 in_channels=3,
                                 hid_channels=[[16, 16, 32], [32, 32, 64]],
                                 time_emb_dim=time_emb_dim)

        self.down2 = PointNetMSG(n_centroids=512,
                                 max_n_points=[16, 32],
                                 radius=[0.1, 0.2],
                                 in_channels=32 + 64 + 3,
                                 hid_channels=[[64, 64, 128], [64, 96, 128]],
                                 time_emb_dim=time_emb_dim)

        self.down3 = PointNetMSG(n_centroids=128,
                                 max_n_points=[16, 32],
                                 radius=[0.2, 0.4],
                                 in_channels=128 + 128 + 3,
                                 hid_channels=[[128, 196, 256], [128, 196, 256]],
                                 time_emb_dim=time_emb_dim)

        self.down4 = PointNetMSG(n_centroids=64,
                                 max_n_points=[16, 32],
                                 radius=[0.4, 0.8],
                                 in_channels=256 + 256 + 3,
                                 hid_channels=[[256, 256, 512], [256, 384, 512]],
                                 time_emb_dim=time_emb_dim)

        self.up1 = FeaturePropagation(512, 1024, [256, 256], time_emb_dim)

        self.up2 = FeaturePropagation(256, 256, [256, 256], time_emb_dim)

        self.up3 = FeaturePropagation(96, 256, [256, 128], time_emb_dim)

        self.up4 = FeaturePropagation(0, 128, [128, 128, 128], time_emb_dim)

        self.predictor = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 3, 1)
        )

    def forward(self, x, time_emb, return_features=False):
        xyz1, feat1 = self.down1(x, time_emb=time_emb)  # b x 96 x 1024
        xyz2, feat2 = self.down2(xyz1, time_emb, point_features=feat1)  # b x 256 x 256
        xyz3, feat3 = self.down3(xyz2, time_emb, point_features=feat2)  # b x 512 x 64
        xyz4, feat4 = self.down4(xyz3, time_emb, point_features=feat3)  # b x 1024 x 16

        feat3 = self.up1(xyz3, xyz4, feat3, feat4, time_emb)  # b x 256 x 64
        feat2 = self.up2(xyz2, xyz3, feat2, feat3, time_emb)  # b x 256 x 256
        feat1 = self.up3(xyz1, xyz2, feat1, feat2, time_emb)  # b x 128 x 1024
        feat0 = self.up4(x, xyz1, None, feat1, time_emb)  # b x 128 x 2048

        if return_features:
            return (feat0, feat1, feat2, feat3), self.predictor(feat0)

        return self.predictor(feat0)
