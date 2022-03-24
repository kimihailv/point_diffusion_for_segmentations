import torch.nn as nn
import torch
from .encoders import PointNetEncoder
from .modules import PointNetMSG, FeaturePropagation, Attention1d


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()

        self.hidden = nn.Conv1d(dim_in, dim_out, 1)
        # self.norm = nn.InstanceNorm1d(dim_out, affine=False)
        self.bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self.scale = nn.Linear(dim_ctx, dim_out)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.InstanceNorm1d(dim_out, affine=False)

    def forward(self, x, ctx):
        scale = self.scale(ctx).unsqueeze(2)
        bias = self.bias(ctx).unsqueeze(2)
        out = self.norm(self.hidden(x)) * scale + bias

        return out


class LatentInjector(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(515, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim_in)
        )

    def forward(self, x, latent):
        latent = self.proj(latent).unsqueeze(2).expand(-1, -1, x.size(2))

        return x + latent


class NoisePredictor(nn.Module):
    def __init__(self, time_dim, residual=True):
        super().__init__()
        self.residual = residual
        # self.encoder = DGCNN()
        self.encoder = PointNetEncoder(512)
        ctx_channels = 256 + time_dim

        '''self.adain = nn.ModuleList([
            ConcatSquashLinear(3, 128, ctx_channels),
            ConcatSquashLinear(128, 256, ctx_channels),
            ConcatSquashLinear(256, 512, ctx_channels),
            ConcatSquashLinear(512, 256, ctx_channels),
            ConcatSquashLinear(256, 128, ctx_channels),
            ConcatSquashLinear(128, 3, ctx_channels)
        ])
        '''

        self.an1 = LatentInjector(3)  # ConcatSquashLinear(3, 32, ctx_channels)
        self.down1 = PointNetMSG(n_centroids=1024,
                                 max_n_points=[16, 32],
                                 radius=[0.1, 0.3],
                                 in_channels=3 + 3,
                                 hid_channels=[[16, 16, 32], [32, 32, 64]])  # b x 96 x 1024

        self.an2 = LatentInjector(96)  # ConcatSquashLinear(96, 96, ctx_channels)
        self.down2 = PointNetMSG(n_centroids=256,
                                 max_n_points=[32, 64],
                                 radius=[0.3, 0.5],
                                 in_channels=96 + 3,
                                 hid_channels=[[64, 64, 128], [64, 96, 128]])  # b x 256 x 256

        self.an3 = LatentInjector(256)  # ConcatSquashLinear(256, 256, ctx_channels)
        self.down3 = PointNetMSG(n_centroids=32,
                                 max_n_points=[64, 128],
                                 radius=[0.5, 0.7],
                                 in_channels=256 + 3,
                                 hid_channels=[[128, 196, 256], [128, 196, 256]])  # b x 512 x 32

        self.attn1 = Attention1d(512, 512, 512)
        self.an4 = LatentInjector(512)  # ConcatSquashLinear(512, 512, ctx_channels)
        self.up1 = FeaturePropagation(256, 512, [256, 256])  # b x 256 x 256

        self.an5 = LatentInjector(256)  # ConcatSquashLinear(256, 256, ctx_channels)
        self.up2 = FeaturePropagation(96, 256, [256, 128, 128])  # b x 128 x 1024

        self.an6 = LatentInjector(128)  # ConcatSquashLinear(128, 128, ctx_channels)
        self.up3 = FeaturePropagation(3, 128, [128, 128, 256])  # b x 128 x 2048

        # 

        self.predictor = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 3, 1)
        )

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, xt, time_emb, return_features, z=None):
        mean = log_var = None

        if z is None:
            ctx, z, mean, log_var = self.get_ctx(x, time_emb)
        else:
            ctx = torch.cat((z, time_emb), dim=1)

        '''shortcut = xt
        for idx, extractor in enumerate(self.adain):
            xt = extractor(xt, ctx)
            if idx != len(self.adain):
                xt = self.lrelu(xt)

        et = xt + shortcut if self.residual else xt
        return et, ctx, (z, mean, log_var)
        '''
        feat0 = self.an1(xt, ctx)
        xyz1, feat1 = self.down1(xt, point_features=feat0)

        feat1 = self.an2(feat1, ctx)
        xyz2, feat2 = self.down2(xyz1, point_features=feat1)

        feat2 = self.an3(feat2, ctx)
        xyz3, feat3 = self.down3(xyz2, point_features=feat2)
        feat3 = self.attn1(feat3, feat3)

        feat3 = self.an4(feat3, ctx)

        feat2 = self.up1(xyz2, xyz3, feat2, feat3)
        feat2 = self.an5(feat2, ctx)

        feat1 = self.up2(xyz1, xyz2, feat1, feat2)
        feat1 = self.an6(feat1, ctx)

        feat0 = self.up3(xt, xyz1, feat0, feat1)

        et = self.predictor(feat0)

        if not return_features:
            return et, ctx, (z, mean, log_var)

        features = [feat0, feat1, feat2, feat3], [xyz1, xyz2, xyz3]

        return features, (et, ctx, (z, mean, log_var))

    def get_ctx(self, x, time_emb):
        # time_emb = self.time_embed(time_emb)
        mean = log_var = None
        if x is not None:
            z, mean, log_var = self.encoder(x)
        else:
            z = torch.randn(time_emb.size(0), 512, device=time_emb.device)
        return torch.cat((z, time_emb), dim=1), z, mean, log_var