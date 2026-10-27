---
title: 'Betikler ve dosyalar'
---

Her betik `--help` ile kullanımını yazar.

## Betikler

| Betik | Ne yapar? |
|---|---|
| [`train_mac.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/train_mac.py) | Değiştirilmemiş `joystick.py` ortamında (düz zemin) Brax PPO; `BerkeleyHumanoidJoystickFlatTerrain` hiperparametreleri; sanal CPU cihazları üzerinde pmap; her checkpoint'te log, ONNX ve parametreler |
| [`onnx_export.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/onnx_export.py) | Brax parametreleri → ONNX, yalnız `onnx` paketiyle (TensorFlow/tf2onnx gerekmez); gürültülü örnekleme için `loc_out`/`scale` çıktıları da var |
| [`smoke.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/smoke.py), [`smoke_pmap.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/smoke_pmap.py) | Ortam derleniyor ve adım atıyor mu, tek cihazda ve D cihazda ne kadar hızlı |
| [`eval_checkpoint.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/eval_checkpoint.py) | Bir ONNX politikasını upstream `MjInfer` ile C MuJoCo'da çalıştırır; mesafe, düşme, ayak adımları; isteğe bağlı video |
| [`eval_grid.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/eval_grid.py) | Aynısı, keşif gürültüsüyle 9 tohum; kaçı düştü, ortalama mesafe, isteğe bağlı 3×3 video |
| [`make_log.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/make_log.py) | Ham log + ölçüm JSON'ları → `train_log.csv`, `ckpt_summary.json`, `reward_curve.png` |
| [`reward_hacking/hile_env.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/hile_env.py) | 2. bölümün ödül dosyası (`VARIANTS`) |
| [`reward_hacking/train_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/train_hile.py) | Ödül varyantlarıyla eğitim, `--restore` ile sıcak başlangıç |
| [`reward_hacking/make_hile_repo.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/make_hile_repo.py) | Open_Duck_Playground kopyasına zeminle çarpışan kütlesiz kutular ekler |
| [`reward_hacking/render_ckpt_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/render_ckpt_hile.py), [`render_grid_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/render_grid_hile.py) | 2. bölümün değerlendiricileri (eğitim fiziği, ek ölçümler) |
| [`sim2real/eval_world.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/sim2real/eval_world.py) | `eval_checkpoint.py` + dünyayı değiştiren ortam değişkenleri (`FORCE`, `FORCE_RAND`, `FLOOR_FRIC`, `MASS_SCALE`) |

## Sonuç klasörleri

| Klasör | İçerik |
|---|---|
| [`results/2026-10-24/`](https://github.com/fport/open-duck-mac/tree/main/results/2026-10-24) | 1. bölüm: `train_log.csv` (tek tablo), `train_log_raw.csv`, `checkpoints.csv`, `ckpt_summary.json`, `train_metrics.jsonl` (PPO kayıpları, parça başına adım/sn), `run_config.json` (PPO parametreleri), `sim/*.json` (her ölçüm), 68,4 M ve 72,4 M ONNX'leri, `params/ckpt_068413440.pkl` (sıcak başlangıç için) |
| [`results/2026-10-24-reward-hacking/`](https://github.com/fport/open-duck-mac/tree/main/results/2026-10-24-reward-hacking) | 2. bölüm: `summary.json`; her varyant için `run.json`, `train_log.csv`, `train_metrics_ep.jsonl`, `onnx/`, `sim/` |

`make_log.py`, 1. bölümün `train_log.csv` ve `ckpt_summary.json` dosyalarını bu dosyalardan bayt bayt yeniden üretir.

## Ortam

- Python 3.12, `requirements-lock.txt` (tam ortam), `requirements.txt` (doğrudan bağımlılıklar: jax 0.6.2, mujoco / mujoco-mjx 3.3.3, brax 0.12.4, playground 0.0.5, onnx, onnxruntime, imageio).
- Open_Duck_Playground commit `b9be205ac64488c23504ca42e5ec790337adeec3`.
- Hazır politika kıyası: Open_Duck_Mini commit `b23317a`'daki `BEST_WALK_ONNX.onnx`.

## Lisans ve teşekkür

- **Open Duck Mini v2** ve **Open_Duck_Playground**: [Antoine Pirrone](https://github.com/apirrone) ve katkıcılar. Robot, MJCF modeli, `joystick.py` ortamı, ödüller, referans hareket (`polynomial_coefficients.pkl`), `mujoco_infer.py` ve hazır `BEST_WALK_ONNX`. Open_Duck_Mini Apache-2.0. Open_Duck_Playground'un kaynak dosyalarında Apache-2.0 başlığı var ama `b9be205`'te depo kökünde LICENSE dosyası yoktu.
- **[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)**, **[MuJoCo / MJX](https://github.com/google-deepmind/mujoco)** ve **[Brax](https://github.com/google/brax)**: Google DeepMind ve katkıcılar, Apache-2.0.
- Bu depodaki kod [Apache License 2.0](https://github.com/fport/open-duck-mac/blob/main/LICENSE) altında. `results/` (loglar, ölçümler, ONNX politikalarımız) da aynı lisansla.
