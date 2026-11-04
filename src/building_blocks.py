"""
Shared building blocks of the paper's model (Fig. 3): the residual self-attention
encoder and the task decoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import N_RAW_FEAT, N_DIST_FEAT

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid_ch = out_ch // 4
        self.conv1 = nn.Conv1d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_ch)
        self.conv3 = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm1d(out_ch))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(),
            nn.Linear(ch // reduction, ch), nn.Sigmoid())
    def forward(self, x):
        y = self.fc(x.mean(dim=2)).view(x.size(0), -1, 1)
        return x * y

class MultiHeadSelfAttention1D(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(out).permute(0, 2, 1)

class ImprovedEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 7, 2, 3, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        self.layer1 = nn.Sequential(ResBlock1D(base_ch, base_ch*2), SEBlock(base_ch*2),
                                     ResBlock1D(base_ch*2, base_ch*2), SEBlock(base_ch*2))
        self.layer2 = nn.Sequential(ResBlock1D(base_ch*2, base_ch*4, 2), SEBlock(base_ch*4),
                                     ResBlock1D(base_ch*4, base_ch*4), SEBlock(base_ch*4))
        self.layer3 = nn.Sequential(ResBlock1D(base_ch*4, base_ch*6, 2), SEBlock(base_ch*6),
                                     ResBlock1D(base_ch*6, base_ch*6), SEBlock(base_ch*6))
        self.layer4 = nn.Sequential(ResBlock1D(base_ch*6, base_ch*8, 2), SEBlock(base_ch*8),
                                     ResBlock1D(base_ch*8, base_ch*8), SEBlock(base_ch*8))
        self.attention = MultiHeadSelfAttention1D(base_ch*8, num_heads=8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.feat_dim = base_ch * 16
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.attention(x)
        return torch.cat([self.gap(x).squeeze(-1), self.gmp(x).squeeze(-1)], dim=1)

class TaskDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev = hd
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PolarityDecoder(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, embed_dim, k, padding=k//2), nn.BatchNorm1d(embed_dim), nn.ReLU())
            for k in [7, 11, 15, 21]])
        self.aggregate = nn.Sequential(
            nn.Conv1d(embed_dim*4, embed_dim*2, 1), nn.BatchNorm1d(embed_dim*2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(embed_dim, 2))
    def forward(self, x):
        fused = torch.cat([c(x) for c in self.convs], dim=1)
        return self.classifier(self.aggregate(fused).squeeze(-1))

class MagnitudeDecoder(nn.Module):
    def __init__(self, feat_dim, dropout=0.3):
        super().__init__()
        self.raw_encoder = nn.Sequential(
            nn.Linear(N_RAW_FEAT, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.dist_encoder = nn.Sequential(
            nn.Linear(N_DIST_FEAT, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + 192, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, enc_feat, raw_feat, dist_feat):
        x = torch.cat([enc_feat, self.raw_encoder(raw_feat), self.dist_encoder(dist_feat)], dim=1)
        return self.fc(x).squeeze(-1)

