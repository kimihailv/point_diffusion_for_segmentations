import torch.nn as nn
import torch
from .modules.attention import Attention
from .modules.shared_mlp import SharedMLP
from .modules.pvconv import PVConv
from .modules.pointnet import PointNetFPModule, PointNetSAModule, PointNetAModule
from functools import partial


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_pointnet2_fp_modules(fp_blocks, time_channels, in_channels, sa_in_channels,
                                with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + time_channels,
                             out_channels=out_channels)
        )
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                with_se=with_se, normalize=normalize, eps=eps)
            for i in range(num_blocks):
                use_attention = i == num_blocks - 1 and voxel_resolution < 16
                fp_blocks.append(block(in_channels,
                                       out_channels,
                                       use_attention=use_attention))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

    return fp_layers, in_channels


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet2_sa_components(sa_blocks, time_channels, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = 3

    sa_layers, sa_in_channels = [], []
    component_idx = 0

    for i, (conv_configs, sa_configs) in enumerate(sa_blocks):
        sa_in_channels.append(in_channels)
        sa_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                with_se=with_se, normalize=normalize, eps=eps)
            for i in range(num_blocks):
                if component_idx != 0 and i == 0:
                    in_channels += time_channels

                use_attn = i == num_blocks - 1 and voxel_resolution < 32
                sa_blocks.append(block(in_channels, out_channels,
                                       use_attention=use_attn))
                in_channels = out_channels

        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            include_coordinates = i != 0
            block = partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                            num_neighbors=num_neighbors,
                            include_coordinates=include_coordinates)

        if conv_configs is None:
            in_channels += time_channels

        sa_blocks.append(block(in_channels=in_channels, out_channels=out_channels))
        in_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

        component_idx += 1

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


class PointNet2(nn.Module):
    def __init__(self, time_emb_dim, sa_blocks, fp_blocks,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks,
            time_channels=time_emb_dim,
            width_multiplier=width_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)
        self.attn = Attention(channels_sa_features, dim=1)

        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks,
            time_channels=time_emb_dim,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, 3],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs, time_emb, return_features):
        coords, features = inputs, inputs
        time_emb = self.time_embed(time_emb).unsqueeze(2)

        coords_list, in_features_list = [], []
        for idx, sa_module in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)

            if idx > 0:
                features = torch.cat((features, time_emb.expand(-1, -1, features.size(2))), dim=1)
            features, coords, _ = sa_module((features, coords, time_emb))

        # in_features_list[0] = inputs[:, 3:, :].contiguous()
        features = self.attn(features)

        for fp_idx, fp_module in enumerate(self.fp_layers):
            features = torch.cat((features, time_emb.expand(-1, -1, features.size(2))), dim=1)
            features, coords, _ = fp_module((coords_list[-1 - fp_idx], coords, features,
                                             in_features_list[-1 - fp_idx], time_emb))

        return self.classifier(features)


class PVCNN2(PointNet2):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]

    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 2, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, time_emb_dim, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            time_emb_dim=time_emb_dim,
            sa_blocks=self.sa_blocks,
            fp_blocks=self.fp_blocks,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
