import torch
import torch.nn as nn
import torch.nn.functional as F
from .Block import Conv3DBlock, Down3D, SpatialDropout3D, FFO3D, SkipProject3D, Block2D, Up2D, CAB


class SVSnet(nn.Module):
    """
    输入: (B, C_in, D, H, W)
    输出: (B, num_classes, H, W)
    """
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 base_channels=(8, 16, 32, 64, 128, 256, 512),
                 use_spatial_dropout=True,
                 dropout_p=0.2,
                 k_depth=4):
        super().__init__()

        C1, C2, C3, C4, C5, C6, C7 = base_channels

        # FFO：多帧3D特征融合 -> 8通道
        self.ffo = FFO3D(in_channels, C1)

        # Encoder 3D
        self.enc1 = Conv3DBlock(C1, C1)
        self.down1 = Down3D(C1)              # 空间下采样
        self.enc2 = Conv3DBlock(C1, C2)

        self.down2 = Down3D(C2)
        self.enc3 = Conv3DBlock(C2, C3)

        self.down3 = Down3D(C3)
        self.enc4 = Conv3DBlock(C3, C4)

        self.down4 = Down3D(C4)
        self.enc5 = Conv3DBlock(C4, C5)

        self.drop5 = SpatialDropout3D(dropout_p) if use_spatial_dropout else nn.Identity()

        self.down5 = Down3D(C5)
        self.enc6 = Conv3DBlock(C5, C6)

        self.drop6 = SpatialDropout3D(dropout_p) if use_spatial_dropout else nn.Identity()

        self.down6 = Down3D(C6)
        self.enc7 = Conv3DBlock(C6, C7)      # 最深层 512

        # Skip projection 3D->2D（Conv3D ks=4x1x1 后 squeeze depth）
        self.proj1 = SkipProject3D(C1, k_depth=k_depth)
        self.proj2 = SkipProject3D(C2, k_depth=k_depth)
        self.proj3 = SkipProject3D(C3, k_depth=k_depth)
        self.proj4 = SkipProject3D(C4, k_depth=k_depth)
        self.proj5 = SkipProject3D(C5, k_depth=k_depth)
        self.proj6 = SkipProject3D(C6, k_depth=k_depth)
        self.proj7 = SkipProject3D(C7, k_depth=k_depth)

        # Decoder 2D：自顶向下（对应紫/黄/蓝）
        # 从最深层开始：HF 通道逐级减半，LF 来自对应的 skip 2D
        self.up6  = Up2D(C7, C6)                # 512 -> 256, 上采样
        self.cab6 = CAB(lf_ch=C6, hf_ch=C6)     # 与 proj6 (256) 融合
        self.dec6 = Block2D(C6, C6)

        self.up5  = Up2D(C6, C5)                # 256 -> 128
        self.cab5 = CAB(lf_ch=C5, hf_ch=C5)
        self.dec5 = Block2D(C5, C5)

        self.up4  = Up2D(C5, C4)                # 128 -> 64
        self.cab4 = CAB(lf_ch=C4, hf_ch=C4)
        self.dec4 = Block2D(C4, C4)

        self.up3  = Up2D(C4, C3)                # 64 -> 32
        self.cab3 = CAB(lf_ch=C3, hf_ch=C3)
        self.dec3 = Block2D(C3, C3)

        self.up2  = Up2D(C3, C2)                # 32 -> 16
        self.cab2 = CAB(lf_ch=C2, hf_ch=C2)
        self.dec2 = Block2D(C2, C2)

        self.up1  = Up2D(C2, C1)                # 16 -> 8
        self.cab1 = CAB(lf_ch=C1, hf_ch=C1)
        self.dec1 = Block2D(C1, C1)

        # 输出层：Conv2D 1x1 + Sigmoid（图中右上角）
        self.head = nn.Conv2d(C1, num_classes, kernel_size=1)
        self.act  = nn.Sigmoid() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        """
        x: (B, C_in, D, H, W)
        """
        # FFO
        x = self.ffo(x)            # (B, 8, D, H, W)

        # Encoder 3D
        e1 = self.enc1(x)          # 8
        d1 = self.down1(e1)
        e2 = self.enc2(d1)         # 16

        d2 = self.down2(e2)
        e3 = self.enc3(d2)         # 32

        d3 = self.down3(e3)
        e4 = self.enc4(d3)         # 64

        d4 = self.down4(e4)
        e5 = self.enc5(d4)         # 128
        e5 = self.drop5(e5)

        d5 = self.down5(e5)
        e6 = self.enc6(d5)         # 256
        e6 = self.drop6(e6)

        d6 = self.down6(e6)
        e7 = self.enc7(d6)         # 512 (deepest)

        # --- 3D -> 2D skip projections ---
        s1 = self.proj1(e1)   # (B, 8,   H, W)
        s2 = self.proj2(e2)   # (B, 16,  H/2, W/2)
        s3 = self.proj3(e3)   # (B, 32,  H/4, W/4)
        s4 = self.proj4(e4)   # (B, 64,  H/8, W/8)
        s5 = self.proj5(e5)   # (B,128,  H/16, W/16)
        s6 = self.proj6(e6)   # (B,256,  H/32, W/32)
        s7 = self.proj7(e7)   # (B,512,  H/64, W/64)  # 依据下采样层数

        # Decoder 2D（顶层开始）
        x = s7                               # (B,512,h/64,w/64)
        x = self.up6(x)                      # -> (B,256, h/32, w/32)
        x = self.cab6(s6, x)                 # 融合 LF=s6, HF=x
        x = self.dec6(x)

        x = self.up5(x)                      # -> 128
        x = self.cab5(s5, x)
        x = self.dec5(x)

        x = self.up4(x)                      # -> 64
        x = self.cab4(s4, x)
        x = self.dec4(x)

        x = self.up3(x)                      # -> 32
        x = self.cab3(s3, x)
        x = self.dec3(x)

        x = self.up2(x)                      # -> 16
        x = self.cab2(s2, x)
        x = self.dec2(x)

        x = self.up1(x)                      # -> 8
        x = self.cab1(s1, x)
        x = self.dec1(x)

        out = self.head(x)                   # (B, num_classes, H, W)
        out = self.act(out)
        return out
