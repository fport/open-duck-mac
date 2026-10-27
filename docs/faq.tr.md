# Sorunlar ve SSS

## Kurulum

??? question "`failed to legalize operation 'mhlo.cholesky'`"
    `jax-metal` (Mac GPU) yüklü ve JAX onu kullanıyor. Bu fizik Metal'de çalışmıyor. `requirements-lock.txt` ile kurulmuş temiz bir `.venv` kullan; `jax-metal` içinde olmamalı.

??? question "İlk import çok uzun sürüyor ya da internetten bir şey indiriyor"
    MuJoCo Playground `mujoco_menagerie`'yi klonlamaya çalışıyor. [Kurulum](setup.md#3-python-ortami) sayfasındaki boş klasör komutunu çalıştır.

??? question "`ModuleNotFoundError: playground`"
    Betiklere ilk argüman olarak Open_Duck_Playground klasörünü verdiğinden emin ol (`external/Open_Duck_Playground`). Betikler `playground` paketini oradan yükler.

??? question "Neden her komutta `python -I`?"
    `-I` (izole mod), Python'un bulunduğun klasörden ya da ortam değişkenlerinden rastgele modül yüklemesini engeller. İndirdiğin bir klasörün içinde çalışırken daha güvenli.

## Eğitim

??? question "Saniyedeki adım sayısı çok düşük"
    `smoke_pmap.py` ile `--devices` için 4, 6, 8 dene. Arka planda ağır işler (video işleme, tarayıcı sekmeleri) hızı yarıya düşürebilir; bizim koşuda parça başına 1.924 ile 7.358 adım/sn arasında oynadı.

??? question "İlk checkpoint'ten sonra bir şey olmuyor gibi"
    İlk JAX derlemesi birkaç dakika sürer. Sonra her parça bizim makinede ~9 dakika. `checkpoints.csv` ve terminaldeki `steps=` satırı ilerlemeyi gösterir.

??? question "Ördek saatlerdir yürümüyor, bir şey mi bozuk?"
    Büyük olasılıkla hayır. Bizimki 11 dakikada ayakta durmayı öğrendi, sonra ~3 saat yerinde saydı ve ancak 68,4 M adımda ayaklarını kaldırdı. Bölüm ödülü ve taklit terimi artıyorsa öğreniyor demektir.

??? question "Eğitimi durdurup devam edebilir miyim?"
    `train_mac.py`'de devam seçeneği yok. `reward_hacking/train_hile.py`'deki `--restore <ckpt.pkl>` bir checkpoint'ten devam eder (politika ve gözlem normalizasyonu; değer ağı sıfırdan).

??? question "Mac uyudu ve eğitim durdu"
    Kapak kapanınca Mac uyur. Uzun koşularda komutun başına `caffeinate -i` ekle.

## Sonuçlar

??? question "Benim sayılarım seninkilerden biraz farklı"
    Değerlendirme C MuJoCo'da, eğitim MJX'te; farklı JAX/MuJoCo sürümleri ya da işlemci küçük sayısal farklar yaratabilir. `requirements-lock.txt` ve sabit commit (`b9be205`) ile en yakın sonucu alırsın. Eğitimin kendisi seed'e ve donanıma göre farklı yollar izleyebilir.

??? question "Bu politikayı gerçek bir Open Duck Mini'de çalıştırabilir miyim?"
    ONNX'ler upstream arayüzünde, ama biz donanımda denemedik. Simdeki motorun gerçeğinden güçlü olduğunu unutma: [3. bölüm](part-3-sim2real.md).

??? question "Microduck ile aynı şey mi?"
    Hayır. Microduck, Hugging Face / Pollen Robotics'in ayrı ürünü. Bu depo Open Duck Mini v2 kullanıyor.
