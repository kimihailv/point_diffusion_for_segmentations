import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, use_bn=True, double_mlp=False):
        super().__init__()
        self.k = k
        norm = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        if double_mlp:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=True),
            )

    def knn(self, x):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def get_graph_feature(self, x, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if not dim9:
                idx = self.knn(x)  # (batch_size, num_points, k)
            else:
                idx = self.knn(x[:, 6:])
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, x):
        x = self.get_graph_feature(x)
        return self.conv(x).max(dim=-1)[0]


class Transform_Net(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm1d(1024) if use_bn else nn.Identity()

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512) if use_bn else nn.Identity()
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256) if use_bn else nn.Identity()

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN(nn.Module):
    N_OUTPUT = 256

    def __init__(self, use_bn=True, k=40):
        super().__init__()
        self.transform_net = Transform_Net(use_bn)
        self.conv1 = EdgeConv(3, 64, k, double_mlp=True, use_bn=use_bn)
        self.conv2 = EdgeConv(64, 64, k, double_mlp=True, use_bn=use_bn)
        self.conv3 = EdgeConv(64, 64, k, use_bn=use_bn)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(64 * 3, 1024, 1, bias=False),
            nn.BatchNorm1d(1024) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024 + 64 * 3, 1024, 1, bias=False),
            nn.BatchNorm1d(1024) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(1024, self.N_OUTPUT, 1, bias=False),
            nn.BatchNorm1d(self.N_OUTPUT) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.N_OUTPUT, self.N_OUTPUT, 1)
        )

        '''self.bottleneck = nn.Sequential(
            nn.Linear(self.N_OUTPUT * 2, self.N_OUTPUT),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.N_OUTPUT, self.N_OUTPUT)
        )'''

        self.mean = nn.Sequential(
            nn.Linear(self.N_OUTPUT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.N_OUTPUT)
        )
        self.log_var = nn.Sequential(
            nn.Linear(self.N_OUTPUT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.N_OUTPUT)
        )

    def forward_features(self, x, return_intermediate=False):
        x0 = self.conv1.get_graph_feature(x)
        t = self.transform_net(x0)
        x = torch.bmm(x.transpose(2, 1), t)
        x = x.transpose(2, 1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x = self.mlp1(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        x = torch.cat((x.expand(-1, -1, x1.size(2)), x1, x2, x3), dim=1)
        x = self.mlp2(x)

        if return_intermediate:
            return x, (x1, x2, x3)

        return x

    def forward(self, x):
        features = self.forward_features(x)
        latent = features.max(dim=2)[0]
        mean = self.mean(latent)
        log_var = self.log_var(latent)

        return self.reparameterize(mean, log_var), mean, log_var

    def reparameterize(self, mean, log_var):
        std = (log_var * 0.5).exp()
        e = torch.randn_like(std)

        return std * e + mean


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return self.reparameterize(m, v), m, v

    def reparameterize(self, mean, log_var):
        std = (log_var * 0.5).exp()
        e = torch.randn_like(std)

        return std * e + mean