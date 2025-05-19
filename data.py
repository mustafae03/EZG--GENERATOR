import os
import torch
import pretty_midi
from torch.utils.data import Dataset

class EzgiVeriSeti(Dataset):
    def __init__(self, klasor_yolu, sekans_uzunlugu=32):
        self.veriler = []
        self.sekans_uzunlugu = sekans_uzunlugu

        for dosya in os.listdir(klasor_yolu):
            if not dosya.endswith(".mid"):
                continue
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(klasor_yolu, dosya))
                if not midi.instruments:
                    continue

                
                notalar = [note.pitch for note in midi.instruments[0].notes]
                if len(notalar) < sekans_uzunlugu + 1:
                    continue

               
                for i in range(0, len(notalar) - sekans_uzunlugu):
                    girdi = notalar[i:i + sekans_uzunlugu]
                    cikti = notalar[i + sekans_uzunlugu]
                    self.veriler.append((girdi, cikti))

            except Exception as e:
                print(f"🚫 Hatalı dosya atlandı: {dosya} ({e})")
                continue

        
        print(f" Toplam kullanılabilir örnek sayısı: {len(self.veriler)}")

    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        girdi, cikti = self.veriler[idx]
        girdi_tensor = torch.tensor(girdi, dtype=torch.float32).unsqueeze(-1)  # (sekans, 1)
        cikti_tensor = torch.tensor(cikti, dtype=torch.long)  # Tek değer
        return girdi_tensor, cikti_tensor
