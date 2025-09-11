import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Basic Block
# ----------------------------

class Conv3DBlock(nn.Module):
    """Block3D: (Conv3D 3x3x3 + BN + ReLU) x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class Down3D(nn.Module):
    """Down3D ks=2x2x2, stride=(1,2,2) + BN + ReLU"""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=(1,2,2), stride=(1,2,2))
        self.bn   = nn.BatchNorm3d(ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SpatialDropout3D(nn.Module):
    """SpatialDropout3D"""
    def __init__(self, p=0.2):
        super().__init__()
        self.drop = nn.Dropout3d(p=p)

    def forward(self, x):
        return self.drop(x)

class FFO3D(nn.Module):
    """Feature Fusion Operation"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 图中只标注了“Conv3D”，这里用 3x3x3 更稳健（可按需改成 1x1x1）
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm3d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SkipProject3D(nn.Module):
    """
    Skip connection
    """
    def __init__(self, ch, k_depth=4):
        super().__init__()
        self.k_depth = k_depth

        self.conv = None
        self.ch   = ch

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        k = min(self.k_depth, D)
        if (self.conv is None) or (self.conv.kernel_size[0] != k):
            self.conv = nn.Conv3d(self.ch, self.ch, kernel_size=(k,1,1), stride=(1,1,1), padding=(0,0,0)).to(x.device)
        x = self.conv(x)
        x = x.mean(dim=2)
        return x

class Block2D(nn.Module):
    """Block2D: (Conv2D 3x3 + BN + ReLU) x1"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Up2D(nn.Module):
    """UpSampling(x2) + Conv2D ks=2x2 + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, padding=0, stride=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.bn(self.conv(x)))
        return x

class CAB(nn.Module):
    """
    Channel Attention Block
    """
    def __init__(self, lf_ch, hf_ch):
        super().__init__()
        self.in_ch = lf_ch + hf_ch
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=1)
        self.relu  = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=1)
        self.sig   = nn.Sigmoid()

        self.split_lf = lf_ch
        self.split_hf = hf_ch

        self.mix = nn.Conv2d(self.in_ch, lf_ch, kernel_size=1)

    def forward(self, lf, hf):

        if lf.shape[2:] != hf.shape[2:]:
            hf = F.interpolate(hf, size=lf.shape[2:], mode="bilinear", align_corners=False)


        # Fusion Attention
        x = torch.cat([lf, hf], dim=1)        # (B, C_lf + C_hf, H, W)
        w = self.gap(x)                       # (B, C, 1, 1)
        w = self.relu(self.fc1(w))
        w = self.sig(self.fc2(w))             # (B, C, 1, 1)

        w_lf = w[:, :self.split_lf]
        w_hf = w[:, self.split_lf:]

        lf_w = lf * w_lf
        hf_w = hf * w_hf

        fused = torch.cat([lf_w, hf_w], dim=1)
        out   = self.mix(fused)
        return out