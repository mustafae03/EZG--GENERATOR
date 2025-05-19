import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modelimiz import EzgiTransformer
from data import EzgiVeriSeti


sekans_uzunlugu = 32
girdi_boyut = 1
model_boyut = 128
sinif_sayisi = 128
batch_size = 64
epoch_sayisi = 20


cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Kullanılan cihaz: {cihaz}")


veri_kumesi = EzgiVeriSeti("augmented", sekans_uzunlugu=sekans_uzunlugu)
data_loader = DataLoader(veri_kumesi, batch_size=batch_size, shuffle=True)


model = EzgiTransformer(girdi_boyut, model_boyut, sinif_sayisi).to(cihaz)
kayip_fonksiyonu = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(epoch_sayisi):
    model.train()
    toplam_kayip = 0

    for girdiler, hedefler in data_loader:
        girdiler, hedefler = girdiler.to(cihaz), hedefler.to(cihaz)
        optimizer.zero_grad()
        cikti = model(girdiler)
        kayip = kayip_fonksiyonu(cikti, hedefler)
        kayip.backward()
        optimizer.step()
        toplam_kayip += kayip.item()

    ort_kayip = toplam_kayip / len(data_loader)
    print(f"📘 Epoch {epoch+1}/{epoch_sayisi} - Ortalama Kayıp: {ort_kayip:.4f}")


torch.save(model.state_dict(), "ezgi_model.pth")
print("💾 Model başarıyla kaydedildi: ezgi_model.pth")
