import torch
import torch.nn as nn
import numpy as np

class KonumsalKodlayici(nn.Module):
    def __init__(self, boyut, max_uzunluk=5000):
        super().__init__()
        pe = torch.zeros(max_uzunluk, boyut)
        pozisyon = torch.arange(0, max_uzunluk, dtype=torch.float).unsqueeze(1)
        bolum_katsayi = torch.exp(torch.arange(0, boyut, 2).float() * (-np.log(10000.0) / boyut))
        pe[:, 0::2] = torch.sin(pozisyon * bolum_katsayi)
        pe[:, 1::2] = torch.cos(pozisyon * bolum_katsayi)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EzgiTransformer(nn.Module):
    def __init__(self, girdi_boyut, gizil_boyut, sinif_sayisi, baslik_sayisi=4, katman_sayisi=2, dropout=0.1):
        super().__init__()
        self.giris_donusum = nn.Linear(girdi_boyut, gizil_boyut)
        self.konumsal_kodlayici = KonumsalKodlayici(gizil_boyut)
        katman = nn.TransformerEncoderLayer(
            d_model=gizil_boyut,
            nhead=baslik_sayisi,
            dropout=dropout,
            batch_first=True  
        )
        self.encoder = nn.TransformerEncoder(katman, num_layers=katman_sayisi)
        self.havuzlama = nn.AdaptiveAvgPool1d(1)
        self.cikis_tahmincisi = nn.Sequential(
            nn.Linear(gizil_boyut, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, sinif_sayisi)
        )

    def forward(self, x):
        x = self.giris_donusum(x)
        x = self.konumsal_kodlayici(x)
        x = self.encoder(x)  
        x = x.permute(0, 2, 1)  
        x = self.havuzlama(x).squeeze(-1)
        return self.cikis_tahmincisi(x)
