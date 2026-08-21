import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

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


class DualGateFusion(nn.Module):
    """Fuse equal-resolution SAM2 and OverLoCK features with independent gates."""

    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels!r}")
        self.channels = channels
        gate_input_channels = channels * 2
        self.sam2_gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.overlock_gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, sam2_feature, overlock_feature):
        if sam2_feature.shape != overlock_feature.shape:
            raise ValueError(
                "DualGateFusion requires identical shapes for SAM2 and OverLoCK "
                f"features, got {tuple(sam2_feature.shape)} and "
                f"{tuple(overlock_feature.shape)}"
            )
        if sam2_feature.ndim != 4:
            raise ValueError(
                "DualGateFusion expects 4D NCHW features, "
                f"got {sam2_feature.ndim} dimensions"
            )
        if sam2_feature.shape[1] != self.channels:
            raise ValueError(
                f"DualGateFusion was configured for {self.channels} channels, "
                f"got {sam2_feature.shape[1]}"
            )

        concatenated = torch.cat((sam2_feature, overlock_feature), dim=1)
        sam2_gate = self.sam2_gate(concatenated)
        overlock_gate = self.overlock_gate(concatenated)
        fused = sam2_feature * sam2_gate + overlock_feature * overlock_gate
        return self.refine(fused)

class InceptionDWConv2d_tiny(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=5, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class InceptionDWConv2d_small(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=7, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class InceptionDWConv2d_middle(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=9, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class InceptionDWConv2d_large(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, skip=False,
                 inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class MFE_module(nn.Module):
    def __init__(self, in_channel):
        super(MFE_module, self).__init__()

        self.DWConv3x3 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=3, groups=in_channel // 4)
        self.DWConv5x5 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=5, groups=in_channel // 4)
        self.DWConv7x7 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=7, groups=in_channel // 4)
        self.PWConv1 = BasicConv2d(in_channel, in_channel, kernel_size=1, act=True)
        self.PWConv2 = BasicConv2d(in_channel, in_channel, kernel_size=1, act=False)
        self.Maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        channels = x.size(1)
        channels_per_part = channels // 4
        x1 = x[:, :channels_per_part, :, :]
        x2 = x[:, channels_per_part:2*channels_per_part, :, :]
        x3 = x[:, 2*channels_per_part:3*channels_per_part, :, :]
        x4 = x[:, 3*channels_per_part:, :, :]
        x1 = self.Maxpool(x1)
        x2 = self.DWConv3x3(x2)
        x3 = self.DWConv5x5(x3)
        x4 = self.DWConv7x7(x4)

        x2 = self.sigmoid(x1) * x2
        x3 = self.sigmoid(x2) * x3
        x4 = self.sigmoid(x3) * x4
        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        x_out = self.PWConv1(x_out)
        x_out = self.PWConv2(x_out)
        x_out = x_out + shortcut
        x_out = self.relu(x_out)

        return x_out

class MFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFE, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.inception1 = InceptionDWConv2d_tiny(in_channel//2)
        self.inception2 = InceptionDWConv2d_small(in_channel//2)
        self.inception3 = InceptionDWConv2d_middle(in_channel//2)
        self.inception4 = InceptionDWConv2d_large(in_channel//2)

        self.incep_dwconv1 = BasicConv2d(in_channel//2, in_channel//2, 1)
        self.incep_dwconv2 = BasicConv2d(in_channel//2, in_channel//2, 1)
        self.incep_dwconv3 = BasicConv2d(in_channel//2, in_channel//2, 1)
        self.incep_dwconv4 = BasicConv2d(in_channel//2, in_channel//2, 1)

        self.mfe1=MFE_module(in_channel//2)
        self.mfe2=MFE_module(in_channel//2)
        self.mfe3=MFE_module(in_channel//2)
        self.mfe4=MFE_module(in_channel//2)

        self.conv1=BasicConv2d(in_channel, in_channel//2, 3, padding=1)
        self.conv2=BasicConv2d(in_channel, in_channel//2, 3, padding=1)
        self.conv3=BasicConv2d(in_channel, in_channel//2, 3, padding=1)
        self.conv4=BasicConv2d(in_channel, in_channel//2, 3, padding=1)

        self.conv5= BasicConv2d(in_channel//2, out_channel, 1)
        self.conv6= BasicConv2d(in_channel//2, out_channel, 1)
        self.conv7= BasicConv2d(in_channel//2, out_channel, 1)
        self.conv8= BasicConv2d(in_channel//2, out_channel, 1)

        self.conv_cat1 = BasicConv2d(out_channel * 4, out_channel, kernel_size=1)
        self.conv_cat2 = BasicConv2d(out_channel, out_channel, kernel_size=1,act=False)
        self.conv_res = BasicConv2d(in_channel, out_channel, kernel_size=1,act=False)

        self.conv_parallel= BasicConv2d(in_channel//2*4, in_channel//2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.inception1(x1)
        x1 = self.incep_dwconv1(x1)
        x1_out = self.mfe1(x1)
        x1 = self.conv5(x1_out)
        
        x2 = self.conv2(x)
        x2 = self.inception2(x2)
        x2 = self.incep_dwconv2(x2)
        x2_out = self.mfe2(x2)
        x2 = self.conv6(x2_out)

        x3 = self.conv3(x)
        x3 = self.inception3(x3)
        x3 = self.incep_dwconv3(x3)
        x3_out = self.mfe3(x3)
        x3 = self.conv7(x3_out)

        x4 = self.conv4(x)
        x4 =self.inception4(x4)
        x4 = self.incep_dwconv4(x4)
        x4_out = self.mfe4(x4)
        x4 = self.conv8(x4_out)

        x_out= torch.cat((x1, x2, x3, x4), 1)
        x_out = self.conv_cat1(x_out)
        x_out = self.conv_cat2(x_out)
        x_out = self.relu(x_out + self.conv_res(x))

        x_parallel = torch.cat((x1_out, x2_out, x3_out, x4_out), 1)
        x_parallel = self.conv_parallel(x_parallel)

        return x_out, x_parallel

class SGA(nn.Module):
    def __init__(self, channel):
        super(SGA, self).__init__()
        self.gatex_conv1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.gateh_conv1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.gatex_conv2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.gateh_conv2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.gatex_conv3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.gateh_conv3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv1 = BasicConv2d(channel, channel//2, kernel_size=1)
        self.conv2 = nn.Conv2d(channel//2, 1, kernel_size=1)

    def forward(self,x0, x1, x2, x3):
        x0_up = F.interpolate(x0, size=x1.size()[2:], mode='bilinear', align_corners=True)

        gatex1 = self.gatex_conv1(torch.cat([x1, x0_up], dim=1))
        gateh1 = self.gateh_conv1(torch.cat([x1, x0_up], dim=1))
        out1 = x1 * gatex1 + x0_up * gateh1
        out1=self.refine1(out1)
        out1 = F.interpolate(out1, size=x2.size()[2:], mode='bilinear', align_corners=True)

        gatex2 = self.gatex_conv2(torch.cat([x2, out1], dim=1))
        gateh2 = self.gateh_conv2(torch.cat([x2, out1], dim=1))
        out2 = x2 * gatex2 + out1 * gateh2
        out2=self.refine2(out2)
        out2 = F.interpolate(out2, size=x3.size()[2:], mode='bilinear', align_corners=True)

        gatex3 = self.gatex_conv3(torch.cat([x3, out2], dim=1))
        gateh3 = self.gateh_conv3(torch.cat([x3, out2], dim=1))
        out3 = x3 * gatex3 + out2 * gateh3
        out3=self.refine3(out3)

        x=self.conv1(out3)
        x=self.conv2(x)

        return x
