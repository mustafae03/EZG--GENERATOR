Ezgi Üreten Yapay Zeka Projesi (Transformer Tabanlı)

Bu projede MIDI formatında müzik verileri kullanılarak ezgi yani melodi üreten bir yapay zeka modeli geliştirildi. Model, sıfırdan yazılmış Transformer mimarisiyle çalışmaktadır.


Projenin Amacı:
Transformer mimarisini kullanarak nota dizileri tahmin eden bir sistem kurmak ve bu sistemin MIDI dosyası olarak ezgi üretmesini sağlamaktır.

Kullanılan Teknolojiler:
- Python
- PyTorch
- pretty_midi
- NumPy
- CUDA

Dosyalar:
- train.py: Modelin eğitildiği ana script
- generate.py: Eğitilen modelle ezgi üreten script
- modelimiz.py: Transformer mimarisi burada tanımlı
- data.py: MIDI dosyalarını okuyup veri seti oluşturur
- gpu_kontrolü.py: CUDA desteğini kontrol ettim
- ezgi_model.pth: Eğitilmiş modelin ağırlıkları
- ornekezgi.mid: Modelin ürettiği örnek ezgi
- ornekezgi.mp3: MIDI’den çevrilmiş ses dosyası 

Kullanım:
1. Eğitim yapmak için:
python train.py

2. Ezgi üretmek için:
python generate.py

Çıktı olarak ornekezgi.mid dosyası oluşur. Bu dosya müzik oynatıcılarla veya MuseScore gibi notasyon yazılımlarıyla açılabilir.

Mustafa Emre Tosuner


