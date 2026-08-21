import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, act=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.channels_single = input_channels // 4
        c = self.channels_single

        # 原来的 channel_reduction 实际不是降维，而是通道混合
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.b1 = BasicConv2d(c, c, kernel_size=3, padding=1, dilation=1)
        self.b2 = BasicConv2d(c, c, kernel_size=3, padding=2, dilation=2)
        self.b3 = BasicConv2d(c, c, kernel_size=5, padding=2, dilation=1)
        self.b4 = BasicConv2d(c, c, kernel_size=5, padding=4, dilation=2)

        self.gate_x1 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )
        self.gate_h1 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )
        self.gate_x2 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )
        self.gate_h2 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )
        self.gate_x3 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )
        self.gate_h3 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )
        self.fusion = BasicConv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, dilation=1, act=True)

    def forward(self, x):
        x = self.channel_mixing(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        p1 = self.b4(x1)

        g_x1 = self.gate_x1(torch.cat([x2, p1], dim=1))
        g_h1 = self.gate_h1(torch.cat([x2, p1], dim=1))
        p2 = self.b3(g_h1 * x2 + g_x1 * p1)

        g_x2 = self.gate_x2(torch.cat([x3, p2], dim=1))
        g_h2 = self.gate_h2(torch.cat([x3, p2], dim=1))
        p3 = self.b2(g_h2 * x3 + g_x2 * p2)

        g_x3 = self.gate_x3(torch.cat([x4, p3], dim=1))
        g_h3 = self.gate_h3(torch.cat([x4, p3], dim=1))
        p4 = self.b1(g_h3 * x2 + g_x3 * p3)

        out = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))

        return out


class CFBR(nn.Module):
    def __init__(self, channel1):
        super(CFBR, self).__init__()
        self.channel1 = channel1
        
        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(self.channel1, 1, 1))
        self.beta  = nn.Parameter(torch.ones(self.channel1, 1, 1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        
        up = y
        input_map = in_map
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        return refine2