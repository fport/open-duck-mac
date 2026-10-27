---
title: "Microduck nedir, nasıl eğitiliyor?"
description: "Pollen Robotics'in 399 dolarlık ördek robotu Microduck'ın donanımı, yazılımı, eğitim tarifi, Open Duck Mini v2 ile farkları ve Mac'te neler yapabileceğin."
---

![Aynı kamera ve ölçekle MuJoCo'da yan yana iki ördek: solda daha küçük, turuncu ayaklı Microduck, sağda daha büyük, sarı ayaklı Open Duck Mini v2](../../assets/microduck-vs-open-duck.jpg)

<sub>Aynı kamera, aynı ölçek, ayakta duruş pozu. Solda Microduck (~25 cm), sağda Open Duck Mini v2 (~42 cm). Bizim render'ımız. Microduck modeli: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl); 3B ağ dosyaları © Pollen Robotics, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Bu görsel de CC BY-NC-SA 4.0 lisanslıdır. Open Duck Mini v2 modeli: Open_Duck_Playground, Apache-2.0.</sub>

:::note[Bu sayfa neye dayanıyor?]
Resmî Pollen sayfaları, `pollen-robotics` GitHub depoları, Hugging Face'teki policy ve Space'ler, basın ve topluluk kaynakları (27 Ekim 2026). Henüz hiçbir Microduck teslim edilmedi; yetenekler satıcının gösterdiği ön üretim birimlerine ait. Kaynaklar birçok noktada çelişiyor; [en sonda](#dikkat-çelişen-bilgiler) topladık.
:::

## Kısaca

Microduck, Hugging Face'e ait **Pollen Robotics**'in 27 Ağustos 2026'da duyurduğu iki ayaklı ördek robot ([Pollen blog](https://pollen-robotics.com/microduck/blog/introducing-microduck/)). Monte edilmiş geliyor, vergi ve kargo hariç **399 $**. Yazılımı açık (Apache-2.0), donanımı kapalı. Davranışları, simülasyonda reinforcement learning (pekiştirmeli öğrenme) ile eğitilmiş küçük neural network'ler.

- **Ne yapıyor (iddia):** kumandayla yürüme, oturup kalkma, tekme, gagayla küçük nesne alma, aksesuarla paten, düşünce kalkma, lazer noktası takibi.
- **Ne yapmıyor:** konuşmuyor; ses tanıma, SLAM, navigasyon ve özerk kavrama henüz yok. Pollen bunları topluluk hedefi olarak tanımlıyor ([The Register](https://www.theregister.com/ai-and-ml/2026/08/27/hugging-face-offers-399-robot-duck-to-help-you-quack-the-ai-code/5293011)).
- **Ne zaman:** mağazadaki güncel uyarıya göre mevcut ön siparişler **Nisan 2027'den sonra** gönderilecek. **Türkiye satış bölgeleri arasında yok** ([Pollen mağazası](https://store.pollen-robotics.com/products/microduck)).

:::caution[Taklit mağazalar]
`microducks.org` ve `microduck.shop` Pollen'ın değil. Resmî satış yalnızca [pollen-robotics.com](https://pollen-robotics.com/microduck/) üzerinden.
:::

## Donanım

| Bileşen | Değer |
|---|---|
| Boy, ağırlık | 25 cm; mağazaya göre 780 g (diğer resmî sayfalar "800 g", "800 g'dan az") |
| Motorlar | 15 adet **Dynamixel XL330**: 14'ü yürüme policy'sinde, 1'i gaga |
| İşlemci | Rockchip RK3566 (4× Cortex-A55), 1 GB RAM, 32 GB depolama, 0,8 TOPS NPU |
| Sensörler | geniş açılı kamera, 8×8 ToF derinlik sensörü, 2 IMU (gövde, kafa), mikrofon, hoparlör, 2 NFC anteni |
| Batarya | çıkarılabilir NP-F550, yaklaşık 1 saat |
| Taşıma | yaklaşık 100 g (Pollen'ın sözlü tahmini, resmî değer değil) |

Kaynaklar: [mağaza](https://store.pollen-robotics.com/products/microduck), [policy-manifest.md](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md), [robotd-design.md](https://github.com/pollen-robotics/microduck/blob/main/docs/design/robotd-design.md), [npu-bringup.md](https://github.com/pollen-robotics/microduck/blob/main/docs/project/npu-bringup.md).

İlginç bir ayrıntı: NPU yürümek için kullanılmıyor. Policy'ler işlemcide ONNX Runtime ile saniyede 50 kez çalışıyor; NPU yalnızca kameradaki küçük bir "ördek dedektörünü" (YOLO11n, INT8) koşturuyor.

## Yazılım: davranış = Hub'daki bir ONNX dosyası

Robotun içinde Rust ile yazılmış birkaç servis (daemon) çalışıyor. Motorlara yalnızca `robotd` dokunuyor; diğer her şey ona JSON-RPC ile **niyet** gönderiyor ([architecture.md](https://github.com/pollen-robotics/microduck/blob/main/docs/design/architecture.md)). Bir yürüme komutu tek satır:

```json
{"jsonrpc":"2.0","method":"robot.move","params":{"vx":0.2,"vy":0.0,"vyaw":0.4}}
```

Her davranış, Hugging Face Hub'da duran bir **ONNX policy dosyası** (`policy.onnx` + `manifest.json`). Tüm policy'ler aynı sözleşmeyi kullanıyor: **61 sayı girer, 14 eklem hedefi çıkar**, saniyede 50 kez. Bu sayede robot yürüme, kalkma ve numara policy'leri arasında anında geçiş yapabiliyor. Resmî policy'ler [`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies) deposunda (Apache-2.0): yürüme, ayakta durma, oturup kalkma, yerden alma, sağ/sol tekme, takla, paten ve yürüme + düşünce kalkma (`velstand`).

Ayrı bir doküman sitesi ve PyPI'da Python SDK'sı yok. Asıl dokümanlar [`microduck/docs/`](https://github.com/pollen-robotics/microduck/tree/main/docs) altındaki Markdown dosyaları; başlamak için en iyi kapı [`cheatsheet.md`](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/cheatsheet.md).

## Nasıl eğitiliyor?

[Kavramlar](/open-duck-mac/concepts/) sayfasındaki her şey burada da geçerli: simülasyonda binlerce kopya, bir reward, PPO. Farklar araçlarda ve reward'da.

| | Microduck ([microduck_rl](https://github.com/pollen-robotics/microduck_rl)) |
|---|---|
| Simülatör | **mjlab** (MuJoCo Warp, GPU'da MuJoCo) |
| Algoritma | PPO (rsl_rl, PyTorch), 4.096 parallel environment |
| Süre | tek CUDA GPU'da kullanılabilir yürüyüş 1–2 saat (README) |
| Ağ | actor ve critic 512-256-128 |
| Reward | **imitation yok**: hız takibi, dik durma, poz, adım başına havada kalma, kafa takibi; kayma, öz çarpışma, sarsıntı cezası |
| Motor modeli | XL330 için **BAM M6**: voltaj kontrol yasası, geri EMK, sürtünme |
| Randomization | kütle ×0,95–1,05, sürtünme 0,9–1,1, ağırlık merkezi ±8 mm'ye kadar, batarya voltajı, voltaj düşüşü, komut gecikmesi, itmeler |
| Backlash | her servoya ±1° backlash (dişli boşluğu), ayrı görev ikizleri |

Kaynak: [microduck_velocity_env_cfg.py](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py), [AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/main/AGENTS.md).

İki karar öğrenen biri için özellikle değerli:

1. **Imitation reward'ı bırakmışlar.** Open Duck Mini v2 bir reference motion'ı taklit ederek öğreniyordu. Microduck yalnız görev reward'larıyla (task reward) öğreniyor. Takla görevinde anahtar kare taklidi denenmiş ve bırakılmış: ara noktalar verilince policy o noktalarda "takılıp kalıyormuş". Bir poz görevi en baştan tek bir hedef veriyor; yolu bulmak RL'in işi.
2. **Gerçekliği motor modeli taşıyor.** Depo, bu ölçekte sim2real farkının büyük kısmının motordan geldiğini söylüyor; motor ideal bir denetleyici yerine voltaj yasasına kadar modelleniyor. Bu, bizim [3. bölümde](/open-duck-mac/part-3-sim2real/) gördüğümüzle aynı ders: Open Duck'ın simdeki motor sınırı katalog değerinin ~%70 üstünde ve sınırı kataloğa indirmek tek başına mesafeyi yarıya indiriyor.

## Open Duck Mini v2 ile farkı

Pollen resmî olarak "Open Duck Mini'den türedi" demiyor. Ama 14 eklem **aynı adlarla ve aynı sırayla** duruyor, araç zinciri aynı (Onshape → onshape-to-robot → MJCF, BAM, 50 Hz, ONNX) ve eğitim kodunun en çok katkı veren yazarı Open Duck Mini'nin yaratıcısı Antoine Pirrone. Doğru çerçeve: aynı ekipten, aynı fikirle, ayrı bir ticari ürün.

| | Open Duck Mini v2 (bu sitedeki ördek) | Microduck |
|---|---|---|
| Boy, ağırlık | ~42 cm | 25 cm, ~780–800 g |
| Servo | Feetech STS3215 | Dynamixel XL330 |
| Bilgisayar | Raspberry Pi Zero 2W, Python | RK3566, Rust |
| Simülasyon + RL | MuJoCo Playground (MJX/JAX) + Brax PPO | mjlab (MuJoCo Warp) + rsl_rl PPO |
| Reward | Disney BDX tarzı imitation + tracking | imitation yok, task reward'lar + curriculum |
| Motor modeli | BAM değerleri sabit kp/sönüm/sürtünmeye aktarılmış | BAM M6 simülasyon döngüsünün içinde |
| Donanım | açık (CAD, BOM, Apache-2.0) | kapalı; ağ dosyaları CC BY-NC-SA |

## Mac'te ne yapabilirsin?

Resmî yığın Mac'i **çalıştırmak ve değerlendirmek** için destekliyor, **eğitmek** için desteklemiyor ([mjlab](https://github.com/mujocolab/mjlab)).

| Basamak | Ne | Resmî mi? | Not |
|---|---|---|---|
| 0. Tarayıcı | Gerçek policy'leri [microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) Space'inde sür | evet | kurulum yok; MuJoCo tarayıcıda (WebAssembly) |
| 1. `infer_policy.py` | Bir ONNX policy'yi işlemcide MuJoCo + BAM ile klavyeden sür | evet | `--no-bam` ile basit motora geç, farkı gör; Mac'te `uv sync` resmî olarak test edilmemiş |
| 2. `duck-sim` | Robotun gerçek Rust yazılımını simüle gövdeye karşı çalıştır | evet, macOS desteği var | Rust, `mjpython`; kamera için Homebrew GStreamer |
| 3. Eğitim | Kendi policy'ni eğit | yalnız CUDA GPU ya da ücretli `--hf-jobs` | Mac CPU için topluluk portları var, doğrulanmadı |

```bash
# Basamak 1 (denemedik; komutlar microduck_rl README'sinden)
git clone https://github.com/pollen-robotics/microduck_rl ~/Pollen/microduck_rl
cd ~/Pollen/microduck_rl
uv run scripts/infer_policy.py --walking <policy>.onnx   # kıyas için: --no-bam
```

Mac CPU'da eğitmek istersen topluluk portları: [microduck-lab](https://github.com/jonathanhawkins/microduck-lab) (CPU MuJoCo + Stable-Baselines3), [microduck-rl-torch](https://github.com/bsprenger/microduck-rl-torch) (PyTorch, Apple MPS), [microduck-mjbatch](https://github.com/tailong-wu/microduck-mjbatch) (toplu CPU MuJoCo), [microdux](https://github.com/noahfarr/microdux) (JAX/MJX). JAX yolunu seçersen bizim karşılaştığımız tuzak geçerli: `jax-metal` Cholesky'yi derleyemiyor, JAX'in **CPU** arka ucunu kullan.

Bir başka not: Microduck Arena'sı Mac'teki sonuçların sunucudakinden farklı çıkacağını açıkça söylüyor; servo modeli birkaç saniyelik simülasyondan sonra ıraksayacak kadar kaotik ([microduck-arena](https://github.com/pollen-robotics/microduck-arena)).

## Dikkat: çelişen bilgiler

| Konu | Ne deniyor | Bizim okumamız |
|---|---|---|
| Teslimat | blog: "Noel 2026'dan önce" · 24 Ekim basın: 4–6 ay · mağaza: "Nisan 2027'den sonra" | en güncel olan mağaza uyarısı |
| Ağırlık | 780 g / 800 g / "800 g'dan az" | 780 g en spesifik resmî değer |
| Servo | resmî depolar XL330 · bir satıcı "Microduck için" Feetech HD-1910 satıyor | HD-1910 klonlarda; STS3215 Open Duck'ın |
| Taşıma | ~100 g · bazı sitelerde "800 g" | 800 g robotun kendi ağırlığı |
| "Açık kaynak" | ürün başlığı "open-source biped" · basın kiti: yalnız yazılım | donanım kapalı |
| Eğitim çerçevesi | resmî depo mjlab · bazı basın Isaac Lab / Newton | resmî hat mjlab |
| GTC sahnesi | TechCrunch kusursuz anlatıyor · Reuters, CNBC: bazı ördekler düştü, komutları kaçırdı | birden çok kaynak aksaklık bildiriyor |

Ayrıntılı, kaynaklı rapor: araştırma notlarımızdan derlendi (27 Ekim 2026). Bir şeyi yanlış ya da eskimiş bulursan [issue aç](https://github.com/fport/open-duck-mac/issues).
