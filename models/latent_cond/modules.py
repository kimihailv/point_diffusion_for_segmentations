import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.mlp = []

        for i, channels in enumerate(hid_channels, start=1):
            self.mlp.append(nn.Conv1d(in_channels, channels, 1))
            self.mlp.append(nn.BatchNorm1d(channels))

            if i < len(hid_channels):
                self.mlp.append(nn.LeakyReLU(negative_slope=0.02))

            in_channels = channels

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


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
    def __init__(self, n_centroids, max_n_points, radius, in_channels, hid_channels):
        super().__init__()
        self.n_centroids = n_centroids
        self.max_n_points = max_n_points
        self.radius = radius
        self.mlps = nn.ModuleList()
        self.attention = nn.ModuleList()

        for channels in hid_channels:
            self.attention.append(AttentionMix(in_channels))
            self.mlps.append(MLP(in_channels, channels))

    def forward(self, xyz, point_features=None):
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
            features = mlp(features)
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
        first_point_idx = idx[:, :, 0:1].expand(batch_size, n_centroids, k).clone()
        first_point_idx.masked_fill_(first_point_idx == n_points, 0)
        mask = idx == n_points
        idx[mask] = first_point_idx[mask]

        # b x n_centroids x k
        return idx, mask


class FeaturePropagation(nn.Module):
    def __init__(self, x_in_channels, y_in_channels, hid_channels):
        super().__init__()
        self.layers = []

        self.attention = Attention1d(x_in_channels + 3, y_in_channels + 3, y_in_channels)
        in_channels = x_in_channels + y_in_channels
        self.mlp = MLP(in_channels, hid_channels)

    def forward(self, x, y, x_features, y_features):
        x = torch.cat((x, x_features), dim=1) if x_features is not None else x
        y = torch.cat((y, y_features), dim=1)

        interpolated = self.attention(x, y)
        if x_features is not None:
            interpolated = torch.cat((interpolated, x_features), dim=1)

        return self.mlp(interpolated)