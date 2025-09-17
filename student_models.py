import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentReconstructiveSubNetwork(nn.Module):
    """
    一個輕量化的重建子網路 (學生模型)。
    採用了類似 U-Net 的編碼器-解碼器結構，但通道數大幅減少。
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(StudentReconstructiveSubNetwork, self).__init__()

        # --- 編碼器部分 (Encoder) ---
        # 假設教師模型的通道數可能是 32, 64, 128, 256
        # 學生模型我們使用 16, 32, 64, 128，減少一半
        self.encoder_conv1 = self._make_conv_block(in_channels, 16)
        self.encoder_conv2 = self._make_conv_block(16, 32)
        self.encoder_conv3 = self._make_conv_block(32, 64)
        self.encoder_conv4 = self._make_conv_block(64, 128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- 瓶頸層 (Bottleneck) ---
        self.bottleneck_conv = self._make_conv_block(128, 256)

        # --- 解碼器部分 (Decoder) ---
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder_conv4 = self._make_conv_block(256 + 128, 128) # 跳躍連接，通道數相加
        self.decoder_conv3 = self._make_conv_block(128 + 64, 64)
        self.decoder_conv2 = self._make_conv_block(64 + 32, 32)
        self.decoder_conv1 = self._make_conv_block(32 + 16, 16)

        # --- 輸出層 ---
        self.output_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _make_conv_block(self, in_ch, out_ch):
        """輔助函數，用來建立一個 (Conv -> BatchNorm -> ReLU) * 2 的區塊"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- 編碼器路徑 ---
        e1_out = self.encoder_conv1(x)
        e2_in = self.pool(e1_out)
        
        e2_out = self.encoder_conv2(e2_in)
        e3_in = self.pool(e2_out)

        e3_out = self.encoder_conv3(e3_in)
        e4_in = self.pool(e3_out)

        e4_out = self.encoder_conv4(e4_in)
        b_in = self.pool(e4_out)

        # --- 瓶頸層 ---
        b_out = self.bottleneck_conv(b_in)
        
        # --- 解碼器路徑 (帶有跳躍連接) ---
        d4_in = self.upsample(b_out)
        d4_in = torch.cat([d4_in, e4_out], dim=1) # 跳躍連接
        d4_out = self.decoder_conv4(d4_in)

        d3_in = self.upsample(d4_out)
        d3_in = torch.cat([d3_in, e3_out], dim=1) # 跳躍連接
        d3_out = self.decoder_conv3(d3_in)

        d2_in = self.upsample(d3_out)
        d2_in = torch.cat([d2_in, e2_out], dim=1) # 跳躍連接
        d2_out = self.decoder_conv2(d2_in)

        d1_in = self.upsample(d2_out)
        d1_in = torch.cat([d1_in, e1_out], dim=1) # 跳躍連接
        d1_out = self.decoder_conv1(d1_in)

        # --- 輸出 ---
        out = self.output_conv(d1_out)
        
        return out


class StudentDiscriminativeSubNetwork(nn.Module):
    """
    一個輕量化的判別/分割子網路 (學生模型)。
    採用一個簡單的前饋卷積網路，層數和通道數都較少。
    """
    def __init__(self, in_channels=6, out_channels=2):
        super(StudentDiscriminativeSubNetwork, self).__init__()

        # 假設教師模型的通道數可能是 128, 256
        # 學生模型我們使用 32, 64，大幅減少
        self.layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # stride=2 用來降採樣
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4 (Upsample back to original size)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Output Layer
            nn.Conv2d(32, out_channels, kernel_size=1) # 輸出 logits，不加 aƒctivation
        )

    def forward(self, x):
        return self.layers(x)