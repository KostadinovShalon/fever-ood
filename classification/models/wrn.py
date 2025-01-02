import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    @staticmethod
    def _make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, godin=False, null_space_red_dim=-1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.godin = godin
        self.null_space_red_dim = null_space_red_dim
        self.nChannels = nChannels[3]
        if not godin:
            if null_space_red_dim <= 0:
                self.fc = nn.Linear(nChannels[3], num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(nChannels[3], null_space_red_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(null_space_red_dim, num_classes))
                self.fc[0].bias.data.zero_()
                self.fc[2].bias.data.zero_()
        else:
            self.g = nn.Sequential(
                nn.Linear(nChannels[3], 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
            # torch.nn.init.xavier_normal_(self.g[0].weight)
            # self.g[0].bias.data = torch.zeros(size=self.g[0].bias.size()).cuda()
            if null_space_red_dim <= 0:
                self.fc = nn.Linear(nChannels[3], num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(nChannels[3], null_space_red_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(null_space_red_dim, num_classes))
                self.fc[0].bias.data.zero_()
                self.fc[2].bias.data.zero_()
            nn.init.kaiming_normal_(self.fc[2].weight.data, nonlinearity="relu")
            self.fc[2].bias.data = torch.zeros(size=self.fc[2].bias.size()).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def penultimate_forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out

    def forward(self, x):
        out = self.penultimate_forward(x)
        return self.fc(out)

    def forward_virtual(self, x):
        out = self.penultimate_forward(x)

        if self.null_space_red_dim > 0:
            out = self.fc[0](out)
            out = self.fc[1](out)
            prob = self.fc[2](out)
        else:
            prob = self.fc(out)
        if self.godin:
            deno = self.g(out)
            return prob / deno, out
        else:
            return prob, out

    def intermediate_forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out
    
    def feature_list(self, x):
        out_list = [] 
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.godin:
            deno = self.g(out)
            return self.fc(out) / deno, out_list
        return self.fc(out), out_list