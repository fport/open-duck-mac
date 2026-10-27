<!-- README.md'nin Türkçe ikizi, 2026-10-27 -->

<div align="center">

[English](README.md) · Türkçe

<img src=".github/assets/hero.jpg" width="100%" alt="MuJoCo'da Open Duck Mini v2, üç kare: düşmüş eğitilmemiş politika, 68,4 M adımda adım atan politikamız ve hazır BEST_WALK_ONNX politikası">

<sub>Solda: 0 adım, keşif gürültüsüyle: düşüyor. Ortada: 68,4 M adımdaki politikamız, ayaklarını kaldıran ilk checkpoint. Sağda: hazır <code>BEST_WALK_ONNX</code> (bizim eğitmediğimiz). Yalnız simülasyon.</sub>

# open-duck-mac

**Open Duck Mini v2'ye bir MacBook'ta, yalnız işlemciyle, sıfırdan yürümeyi öğretiyoruz.<br>Sonra onu bozup neden bozulduğunu ölçüyoruz.**

[![Simülasyon: MuJoCo MJX](https://img.shields.io/badge/sim%C3%BClasyon-MuJoCo_MJX-FFAF00)](https://github.com/google-deepmind/mujoco)
[![Apple Silicon, yalnız CPU](https://img.shields.io/badge/Apple_Silicon-yaln%C4%B1z_CPU-111111?logo=apple&logoColor=white)](https://fport.github.io/open-duck-mac/part-1-walking/#neden-cpu)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Doküman: TR / EN](https://img.shields.io/badge/dok%C3%BCman-TR_%2F_EN-FF7A3D)](https://fport.github.io/open-duck-mac/)
[![YouTube @getporti](https://img.shields.io/badge/YouTube-@getporti-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@getporti)
[![Lisans: Apache-2.0](https://img.shields.io/badge/lisans-Apache--2.0-05071A)](LICENSE)

[Doküman](https://fport.github.io/open-duck-mac/) · [Kurulum](https://fport.github.io/open-duck-mac/setup/) · [Kavramlar](https://fport.github.io/open-duck-mac/concepts/) · [Deneyler](https://fport.github.io/open-duck-mac/part-1-walking/) · [Sen ne yapabilirsin?](https://fport.github.io/open-duck-mac/experiments/) · [Videolar](https://www.youtube.com/@getporti)

</div>

---

[Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini), bir Apple Silicon Mac'te [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) ortamında PPO ile yürümeyi öğreniyor: işlemcide JAX, MJX fiziği, Brax PPO. Robot, ortam ve ödüller Antoine Pirrone'un; hiç değiştirilmedi. Bizim olanlar: Mac için bir eğitim betiği, TensorFlow gerektirmeyen bir ONNX dönüştürücü, değerlendiriciler ve videolarımızın arkasındaki her koşunun logları.

[Doküman sitesi](https://fport.github.io/open-duck-mac/) (Türkçe ve İngilizce) seni boş bir Mac'ten kendi deneylerine götürür: kurulum, matematiksiz kavramlar, her deney, deneyebileceğin fikirler.

## Videolar

[YouTube @getporti](https://www.youtube.com/@getporti)'de Türkçe kısa videolar. Her biri bu depodaki bir deney.

| | Video | Deney |
|---|---|---|
| Ördek 1 | [Viral ördeği almadım, kardeşini Mac'imde eğittim](https://youtube.com/shorts/gfCUTIHldt0) | [1. Yürümeyi öğrenmek](https://fport.github.io/open-duck-mac/part-1-walking/) |
| Ördek 2 | "Ördeğe 'hızlı ol' dedim. Yüzüstü sürünmeyi buldu." (28 Ekim 2026) | [2. Ödül hilesi](https://fport.github.io/open-duck-mac/part-2-reward-hacking/) |
| Ördek 3 | yakında: simülasyon yalan söyler | [3. Sim2real](https://fport.github.io/open-duck-mac/part-3-sim2real/) (devam ediyor) |

## Sonuçlar

| | |
|---|---|
| Yürümeyi öğrenmek | M5 MacBook'ta **3,57 saatte 72,4 M adım** (6 işlemci çekirdeği, 1.536 paralel ördek). ~11 dakikada ayağa kalktı, ~3 saat yerinde saydı, **68,4 M** adımda ilk kez ayaklarını kaldırdı: 5 sn'de 29,8 cm (hazır politika: 58,5 cm). |
| Ödül hilesi | Aynı politika, yalnız ödül değişti. "Hızlı ol" → yüzüstü sürünüyor (14,9 dk). "Enerji harcama" → donuyor (7,5 dk). "Ayaklarını kaldır" → yere yatıp ayaklarını havaya dikiyor (30 dk). |
| Sim2real | Simdeki motor (±3,23 N·m), gerçek STS3215 servodan (~1,86 N·m) %70–75 daha güçlü. Gerçek sınırda aynı politika **yarı mesafe** gidiyor; kaygan zemin ve ek ağırlık neredeyse hiç etkilemiyor. |

Her biri tek seed, tek koşu; yalnız simülasyon, gerçek robot yok. Mac'in GPU'su MJX'i çalıştıramıyor (`jax-metal`'da `cholesky` yok), bu yüzden her şey işlemcide; işlemciyi tek iş parçacıklı 6 sanal cihaza bölmek hızı iki katına çıkarıyor.

## Hızlı başlangıç

```bash
git clone https://github.com/fport/open-duck-mac.git && cd open-duck-mac
git clone https://github.com/apirrone/Open_Duck_Playground.git external/Open_Duck_Playground
git -C external/Open_Duck_Playground checkout b9be205ac64488c23504ca42e5ec790337adeec3

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
mkdir -p "$(.venv/bin/python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("mujoco_playground").origin))')/external_deps/mujoco_menagerie"

# eğittiğimiz politikayı ölç ("dx_m": 0.2882 yazar)
.venv/bin/python -I scripts/eval_checkpoint.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/c72.mp4

# eğit (bizim tam koşumuz; 10 M adımdan, ≈ 30 dk, sonra ayakta duruyor)
nice -n 10 .venv/bin/python -I scripts/train_mac.py external/Open_Duck_Playground runs/my-run \
  --devices 6 --envs_per_dev 256 --num_timesteps 72000000 --num_evals 19 --seed 0 \
  --render_script scripts/eval_checkpoint.py
```

Adım adım kurulum, sorun giderme ve tüm seçenekler: **[doküman](https://fport.github.io/open-duck-mac/setup/)**.

## Depo

```
scripts/                 train_mac.py, onnx_export.py, eval_checkpoint.py, eval_grid.py, make_log.py, smoke*.py
scripts/reward_hacking/  2. bölüm: ödül dosyası (hile_env.py), eğitim betiği, çarpışma kutusu yaması, değerlendiriciler
scripts/sim2real/        3. bölüm: eval_world.py (tork, sürtünme, kütle değiştir)
results/2026-10-24/      1. bölüm: loglar, checkpoint tablosu, her ölçüm, ONNX politikaları, sıcak başlangıç parametreleri
results/2026-10-24-reward-hacking/   2. bölüm: dört varyantın her logu ve ölçümü
docs/                    doküman sitesi (MkDocs, Türkçe + İngilizce)
```

## Teşekkür ve lisans

**Open Duck Mini v2** ve **Open_Duck_Playground**: [Antoine Pirrone](https://github.com/apirrone) ve katkıcılar. Robot, model, ortam, ödüller, referans hareket ve hazır `BEST_WALK_ONNX` onların. [MuJoCo / MJX](https://github.com/google-deepmind/mujoco), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) ve [Brax](https://github.com/google/brax): Google DeepMind ve katkıcılar. Bunun üzerine bir şey kurarsan önce Open Duck Mini'yi an: robot da ortam da onların.

Kod ve `results/` [Apache License 2.0](LICENSE) altında. Tam teşekkür listesi: [doküman](https://fport.github.io/open-duck-mac/reference/).
