---
title: 'Sen ne yapabilirsin?'
---

![Sarı bir halkanın içinde birbirini iten beş Open Duck Mini; bazıları devrilmiş](../../assets/duck-arena.jpg)

<sub>Aynı sahnede beş ördek, her biri hazır `BEST_WALK_ONNX` ile bağımsız yürüyor ve birbirini itiyor. Gerçek MuJoCo fiziği, yazılmış hareket yok.</sub>

Kendi Mac'inde deneyebileceğin fikirler, kolaydan zora. Süreler bizim M5'teki hıza (≈5.600 adım/sn) göre kaba tahmin. "Denedik" yazmayanları biz de denemedik; sonucunu bilmiyoruz.

:::tip[Bir deneyi paylaşmak istersen]
Sonucunu bir issue ya da pull request olarak aç: komut, seed, süre ve `run.json` / ölçüm JSON'u yeter. Videolarda kullanırsak adınla anarız.
:::

## Kolay (dakikalar – 1 saat)

### 1. Dünyayı boz, policy'yi ölç · 5 dk · denedik

Eğitim yok, yalnız ölçüm. [`eval_world.py`](/open-duck-mac/part-3-sim2real/#deney-1-yeniden-eğitmeden-farklı-dünyalar) ile tork, sürtünme ya da kütleyi değiştir. Soru: ördek hangi değişikliğe en duyarlı? Kütle ×1,5'te ne olur? Sürtünme 0,2'de?

### 2. Farklı hızda yürüt · 5 dk

`eval_checkpoint.py --vx 0.3` (iki kat hızlı komut) ya da `--vx 0` (dur). Eğitimde komut rastgele seçildiği için policy farklı hızları biliyor olmalı. Mesafe komutla birlikte artıyor mu?

### 3. Kendi reward'ını yaz · 20–40 dk · denedik

[2. bölümdeki](/open-duck-mac/part-2-reward-hacking/#kendi-rewardını-yaz) gibi `VARIANTS`'a bir satır ekle. Fikirler: "sessiz ol" (`action_rate` −5), "taklidi unut" (`imitation` 0, gerisi resmî), "geri git" (`fwd_vel` −10). Ördeğin bulduğu hileyi tahmin et, sonra bak.

### 4. Başka bir seed · 30 dk (10 M) – 3,6 sa (72 M)

`train_mac.py --seed 1`. Bizim ördek ayaklarını 68,4 M'de kaldırdı; seninki ne zaman kaldıracak? Bu, "tek koşu istatistik değildir" uyarısının neden önemli olduğunu gösterir.

## Orta (birkaç saat)

### 5. Resmî tarifin sonuna kadar eğit · ~7,5 sa

`--num_timesteps 150000000`. Biz %48'inde durduk. 150 M'de `BEST_WALK_ONNX`'in 58,5 cm'ine yaklaşıyor mu? Gecelik bir koşu.

### 6. Zayıf motorla eğit · 1–2 sa

[3. bölümdeki](/open-duck-mac/part-3-sim2real/) deney: tork sınırını katalog değerine indir ya da rastgele yap, `scripts/sim2real/train_sim2real.py --force 1.86` veya `--force_rand 1.40 1.90`. Bizde 72 M step yetmedi; farklı bir tohumla ya da sabit 1,86 ile dene.

### 7. Engebeli zemin · birkaç saat

Open_Duck_Playground'da `rough_terrain_backlash` görevi var (sabit commit'te yalnız backlash'li sürümü). `train_mac.py`'de `joystick.Joystick(task="flat_terrain")` satırını değiştirmen gerekir. Düz zeminde eğitilmiş ördek engebede ne yapıyor, engebede eğitilen düzde ne yapıyor?

## Zor (proje)

### 8. Düşünce kalkmayı öğret

Hugging Face'in Microduck'ı düşünce kalkabiliyor; Open Duck environment'ında kalkma görevi yok. Gerekenler: yerde başlayan episode'lar, gövde yüksekliği reward'ı ve zeminle çarpışan gövde (2. bölümdeki [çarpışma kutuları](/open-duck-mac/part-2-reward-hacking/#çarpışma-kutuları) hazır). soyeon24'ün açık döngü denemesi çömelmede kalmış (15 cm duruşun 10,6 cm'si). Kalkmak yürümekten zor.

### 9. Bir gösteri hareketi öğret (Disney tarzı)

Disney'in BDX robotu yürümenin yanında "mutlu dans", "zıplama", "öfke nöbeti" gibi kısa hareketleri de imitation reward ile öğreniyor ([arXiv 2501.05204](https://arxiv.org/abs/2501.05204); tek RTX 4090'da ~22 gün). Küçük sürümü: kendi anahtar karelerinden bir reference motion (referans hareket) üret ([Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator)) ve imitation reward'ı ona bağla.

### 10. Gerçek robota taşı

ONNX'lerimiz upstream arayüzünde (`obs` 1×101 → `continuous_actions`), yani [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) ile çalışması beklenir. Biz donanımda denemedik. Önce 6. deneyi yap: simde 3,23 N·m ile eğitilmiş policy katalog değeri 1,86 N·m'de yarı mesafe gidiyor; gerçek servonun hangisine yakın olduğunu bilmiyoruz.
