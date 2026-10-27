---
title: "Kurulum: boş bir Mac'ten ilk eğitime"
---

Bu sayfa, hiçbir şey kurulu olmayan bir Apple Silicon Mac'ten başlayıp ördeğin ilk eğitim koşusuna kadar her adımı anlatır. Biz yalnız macOS'ta denedik. Linux işlemcilerde de çalışması beklenir ama test etmedik.

## Neye ihtiyacın var?

| | |
|---|---|
| Bilgisayar | Apple Silicon Mac (M1 ve sonrası). Bizimki M5, 10 çekirdek. |
| Disk | Birkaç GB (Python ortamı + checkpoint'ler). |
| Zaman | Kurulum ~15 dk. Tam eğitim bizim makinede 3,6 saat; ilk sonuçlar 10–15 dakikada görünür. |
| Bilgi | Terminalde komut yapıştırabilmek yeter. Python bilmek işini kolaylaştırır ama şart değil. |

:::note[Neden GPU değil?]
Mac'in GPU'su (Metal) bu fiziği çalıştıramıyor: `jax-metal`, MJX'in ihtiyaç duyduğu `cholesky` işlemini desteklemiyor (`failed to legalize operation 'mhlo.cholesky'`). Bu bir donanım sınırı değil, eksik yazılım desteği. Bu yüzden her şey işlemcide. Ayrıntı: [1. Yürümeyi öğrenmek](/open-duck-mac/part-1-walking/#neden-cpu).
:::

## 1. Temel araçlar

```bash
# Git ve derleyici araçları (yüklüyse "already installed" der)
xcode-select --install

# uv: hızlı Python ve paket yöneticisi
curl -LsSf https://astral.sh/uv/install.sh | sh
```

İsteğe bağlı: 9 ördeklik ızgara videosu için `ffmpeg` (`brew install ffmpeg`). Yoksa betikler yalnız ölçüm yazar ve mozaiği atlar.

## 2. Bu depoyu ve ortamı indir

```bash
git clone https://github.com/fport/open-duck-mac.git
cd open-duck-mac

git clone https://github.com/apirrone/Open_Duck_Playground.git external/Open_Duck_Playground
git -C external/Open_Duck_Playground checkout b9be205ac64488c23504ca42e5ec790337adeec3
```

`b9be205` (5 Ağustos 2025, "fix onnx export") bizim eğittiğimiz commit. Sonuçları birebir tekrar etmek istiyorsan bu commit'te kal. `external/` klasörü `.gitignore`'da; depoya girmez.

## 3. Python ortamı

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

`requirements-lock.txt` bizim koşumuzun birebir ortamı; `requirements.txt` yalnız doğrudan bağımlılıkları listeler.

MuJoCo Playground ilk import'ta koca bir robot kütüphanesini (`mujoco_menagerie`) klonlamaya çalışır; bu robot için gerekmez. Boş bir klasör bu klonu atlatır:

```bash
mkdir -p "$(.venv/bin/python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("mujoco_playground").origin))')/external_deps/mujoco_menagerie"
```

## 4. Duman testi (1–2 dakika)

```bash
.venv/bin/python -I scripts/smoke.py external/Open_Duck_Playground --envs 64
.venv/bin/python -I scripts/smoke_pmap.py external/Open_Duck_Playground --devices 6 --envs_per_dev 256
```

İlki ortamın derlenip adım attığını gösterir. İkincisi işlemciyi D sanal cihaza böler ve saniyedeki adım sayısını yazar. `--devices`'ı 4, 6, 8 deneyip en hızlı olanı seç, birkaç çekirdeği boş bırak. Bizim makinede: 1 cihaz × 1.024 environment = 2.368 adım/sn; 6 × 256 = 4.621 adım/sn.

## 5. Eğitmeden önce dene: hazır policy'lerimiz

Eğitim saatler sürer. Önce bizim eğittiğimiz policy'yi ölç:

```bash
.venv/bin/python -I scripts/eval_checkpoint.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/c72.mp4
```

Çıktıda `"dx_m": 0.2882` görmelisin (5 sn'de 28,8 cm). `out/c72.mp4` ördeğin videosu. Yalnız sayı istiyorsan `--no_video` ekle.

## 6. İlk eğitim

```bash
nice -n 10 .venv/bin/python -I scripts/train_mac.py external/Open_Duck_Playground runs/ilk \
  --devices 6 --envs_per_dev 256 --num_timesteps 10000000 --num_evals 4 --seed 0 \
  --render_script scripts/eval_checkpoint.py
```

10 M adım bizim makinede ~30 dakika. Ördek bu sürede ayakta durmayı öğrenir ama henüz yürümez; yürümeye başlaması için 70 M civarı gerekti. Bizim tam koşumuz:

```bash
nice -n 10 .venv/bin/python -I scripts/train_mac.py external/Open_Duck_Playground runs/my-run \
  --devices 6 --envs_per_dev 256 --num_timesteps 72000000 --num_evals 19 --seed 0 \
  --render_script scripts/eval_checkpoint.py
```

Her parçanın (~9 dk) sonunda `runs/<ad>/ckpt/` altına bir `.onnx` ve `.pkl`, `checkpoints.csv`'ye bir satır ve (`--render_script` ile) arka planda 5 sn'lik bir video + ölçüm gelir. İşlemci yetmiyorsa `--render_script`'i çıkar.

:::tip[Bilgisayarı kullanmaya devam et]
`nice -n 10` eğitime düşük öncelik verir; tarayıcı ve editör akıcı kalır. Kapakı kapatmak uykuya sokar ve eğitimi durdurur; uzun koşularda `caffeinate -i` ile çalıştırabilirsin.
:::

## 7. Grafik ve log

```bash
.venv/bin/python scripts/make_log.py runs/my-run
```

`train_log.csv`, `ckpt_summary.json` ve `reward_curve.png` üretir. Neyi okuyacağını [Kavramlar](/open-duck-mac/concepts/) ve [1. bölüm](/open-duck-mac/part-1-walking/) anlatıyor.

Bir şey ters gittiyse: [Sorunlar ve SSS](/open-duck-mac/faq/).
