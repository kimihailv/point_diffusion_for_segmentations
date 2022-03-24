import torch.nn as nn
from . import functional as F
from .voxelization import Voxelization
from .shared_mlp import SharedMLP
from .attention import Attention
from .se import SE3d


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, resolution, use_attention,
                 with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(8, out_channels),
            Attention(out_channels) if use_attention else nn.SiLU()
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, time_emb = inputs

        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, time_emb
