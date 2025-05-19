import torch
from modelimiz import EzgiTransformer
import pretty_midi
import numpy as np
import os


sekans_uzunlugu = 32
girdi_boyut = 1
model_boyut = 128
sinif_sayisi = 128
uretilecek_adim = 100


cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Üretimde kullanılan cihaz: {cihaz}")


model = EzgiTransformer(girdi_boyut, model_boyut, sinif_sayisi).to(cihaz)
if not os.path.exists("ezgi_model.pth"):
    raise FileNotFoundError(" Model dosyası bulunamadı: ezgi_model.pth")

model.load_state_dict(torch.load("ezgi_model.pth", map_location=cihaz))
model.eval()

baslangic = list(np.random.randint(60, 72, size=sekans_uzunlugu))
uretilen = baslangic.copy()

with torch.no_grad():
    for _ in range(uretilecek_adim):
        girdi = torch.tensor(uretilen[-sekans_uzunlugu:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(cihaz)
        cikti = model(girdi)
        olasilik = torch.softmax(cikti, dim=-1).squeeze().cpu().numpy()
        tahmin = int(np.random.choice(len(olasilik), p=olasilik))

        uretilen.append(tahmin)

midi = pretty_midi.PrettyMIDI()
instr = pretty_midi.Instrument(program=0)

zaman = 0.0
sure = 0.5
for pitch in uretilen:
    if 0 <= pitch <= 127:
        nota = pretty_midi.Note(velocity=100, pitch=int(pitch), start=zaman, end=zaman + sure)
        instr.notes.append(nota)
    zaman += sure

midi.instruments.append(instr)
midi.write("uretildi.mid")
print(" mıdı dosyası başarıyla oluşturuldu: uretildi.mid")
