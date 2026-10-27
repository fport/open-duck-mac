---
title: 'Sorunlar ve SSS'
---

## Kurulum

<details>
<summary><code>failed to legalize operation 'mhlo.cholesky'</code></summary>

`jax-metal` (Mac GPU) yüklü ve JAX onu kullanıyor. Bu fizik Metal'de çalışmıyor. `requirements-lock.txt` ile kurulmuş temiz bir `.venv` kullan; `jax-metal` içinde olmamalı.

</details>

<details>
<summary>İlk import çok uzun sürüyor ya da internetten bir şey indiriyor</summary>

MuJoCo Playground `mujoco_menagerie`'yi klonlamaya çalışıyor. [Kurulum](/open-duck-mac/setup/#3-python-ortamı) sayfasındaki boş klasör komutunu çalıştır.

</details>

<details>
<summary><code>ModuleNotFoundError: playground</code></summary>

Betiklere ilk argüman olarak Open_Duck_Playground klasörünü verdiğinden emin ol (`external/Open_Duck_Playground`). Betikler `playground` paketini oradan yükler.

</details>

<details>
<summary>Neden her komutta <code>python -I</code>?</summary>

`-I` (izole mod), Python'un bulunduğun klasörden ya da ortam değişkenlerinden rastgele modül yüklemesini engeller. İndirdiğin bir klasörün içinde çalışırken daha güvenli.

</details>

## Eğitim

<details>
<summary>Saniyedeki adım sayısı çok düşük</summary>

`smoke_pmap.py` ile `--devices` için 4, 6, 8 dene. Arka planda ağır işler (video işleme, tarayıcı sekmeleri) hızı yarıya düşürebilir; bizim koşuda parça başına 1.924 ile 7.358 adım/sn arasında oynadı.

</details>

<details>
<summary>İlk checkpoint'ten sonra bir şey olmuyor gibi</summary>

İlk JAX derlemesi birkaç dakika sürer. Sonra her parça bizim makinede ~9 dakika. `checkpoints.csv` ve terminaldeki `steps=` satırı ilerlemeyi gösterir.

</details>

<details>
<summary>Ördek saatlerdir yürümüyor, bir şey mi bozuk?</summary>

Büyük olasılıkla hayır. Bizimki 11 dakikada ayakta durmayı öğrendi, sonra ~3 saat yerinde saydı ve ancak 68,4 M adımda ayaklarını kaldırdı. Episode reward'ı ve imitation terimi artıyorsa öğreniyor demektir.

</details>

<details>
<summary>Eğitimi durdurup devam edebilir miyim?</summary>

`train_mac.py`'de devam seçeneği yok. `reward_hacking/train_hile.py`'deki `--restore <ckpt.pkl>` bir checkpoint'ten devam eder (policy ve observation normalization; value network sıfırdan).

</details>

<details>
<summary>Mac uyudu ve eğitim durdu</summary>

Kapak kapanınca Mac uyur. Uzun koşularda komutun başına `caffeinate -i` ekle.

</details>

## Sonuçlar

<details>
<summary>Benim sayılarım seninkilerden biraz farklı</summary>

Değerlendirme C MuJoCo'da, eğitim MJX'te; farklı JAX/MuJoCo sürümleri ya da işlemci küçük sayısal farklar yaratabilir. `requirements-lock.txt` ve sabit commit (`b9be205`) ile en yakın sonucu alırsın. Eğitimin kendisi seed'e ve donanıma göre farklı yollar izleyebilir.

</details>

<details>
<summary>Bu policy'yi gerçek bir Open Duck Mini'de çalıştırabilir miyim?</summary>

ONNX'ler upstream arayüzünde, ama biz donanımda denemedik. Simdeki motor sınırı servonun katalog değerinin ~%70 üstünde: [3. bölüm](/open-duck-mac/part-3-sim2real/).

</details>

<details>
<summary>Microduck ile aynı şey mi?</summary>

Hayır. Microduck, Hugging Face'e ait Pollen Robotics'in ayrı ürünü; bu depo Open Duck Mini v2 kullanıyor. Farklar ve Mac'te neler yapılabileceği: [Microduck](/open-duck-mac/microduck/).

</details>
