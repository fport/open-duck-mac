# Sen ne yapabilirsin?

Kendi Mac'inde deneyebileceğin fikirler, kolaydan zora. Süreler bizim M5'teki hıza (≈5.600 adım/sn) göre kaba tahmin. "Denedik" yazmayanları biz de denemedik; sonucunu bilmiyoruz.

!!! tip "Bir deneyi paylaşmak istersen"
    Sonucunu bir issue ya da pull request olarak aç: komut, seed, süre ve `run.json` / ölçüm JSON'u yeter. Videolarda kullanırsak adınla anarız.

## Kolay (dakikalar – 1 saat)

### 1. Dünyayı boz, politikayı ölç · 5 dk · denedik

Eğitim yok, yalnız ölçüm. [`eval_world.py`](part-3-sim2real.md#deney-1) ile tork, sürtünme ya da kütleyi değiştir. Soru: ördek hangi değişikliğe en duyarlı? Kütle ×1,5'te ne olur? Sürtünme 0,2'de?

### 2. Farklı hızda yürüt · 5 dk

`eval_checkpoint.py --vx 0.3` (iki kat hızlı komut) ya da `--vx 0` (dur). Eğitimde komut rastgele seçildiği için politika farklı hızları biliyor olmalı. Mesafe komutla birlikte artıyor mu?

### 3. Kendi ödülünü yaz · 20–40 dk · denedik

[2. bölümdeki](part-2-reward-hacking.md#kendi-odulunu-yaz) gibi `VARIANTS`'a bir satır ekle. Fikirler: "sessiz ol" (`action_rate` −5), "taklidi unut" (`imitation` 0, gerisi resmî), "geri git" (`fwd_vel` −10). Ördeğin bulduğu hileyi tahmin et, sonra bak.

### 4. Başka bir seed · 30 dk (10 M) – 3,6 sa (72 M)

`train_mac.py --seed 1`. Bizim ördek ayaklarını 68,4 M'de kaldırdı; seninki ne zaman kaldıracak? Bu, "tek koşu istatistik değildir" uyarısının neden önemli olduğunu gösterir.

## Orta (birkaç saat)

### 5. Resmî tarifin sonuna kadar eğit · ~7,5 sa

`--num_timesteps 150000000`. Biz %48'inde durduk. 150 M'de `BEST_WALK_ONNX`'in 58,5 cm'ine yaklaşıyor mu? Gecelik bir koşu.

### 6. Zayıf motorla eğit · 1–2 sa

[3. bölümdeki](part-3-sim2real.md) deney: tork sınırını gerçek servoya indir ya da rastgele yap. Eğitim betiği bölüm tamamlanınca depoya eklenecek.

### 7. Engebeli zemin · birkaç saat

Open_Duck_Playground'da `rough_terrain_backlash` görevi var (sabit commit'te yalnız backlash'li sürümü). `train_mac.py`'de `joystick.Joystick(task="flat_terrain")` satırını değiştirmen gerekir. Düz zeminde eğitilmiş ördek engebede ne yapıyor, engebede eğitilen düzde ne yapıyor?

## Zor (proje)

### 8. Düşünce kalkmayı öğret

Hugging Face'in Microduck'ı düşünce kalkabiliyor; Open Duck ortamında kalkma görevi yok. Gerekenler: yerde başlayan bölümler, gövde yüksekliği ödülü ve zeminle çarpışan gövde (2. bölümdeki [çarpışma kutuları](part-2-reward-hacking.md#carpisma-kutulari) hazır). soyeon24'ün açık döngü denemesi çömelmede kalmış (15 cm duruşun 10,6 cm'si). Kalkmak yürümekten zor.

### 9. Bir gösteri hareketi öğret (Disney tarzı)

Disney'in BDX robotu yürümenin yanında "mutlu dans", "zıplama", "öfke nöbeti" gibi kısa hareketleri de taklit ödülüyle öğreniyor ([arXiv 2501.05204](https://arxiv.org/abs/2501.05204); tek RTX 4090'da ~22 gün). Küçük sürümü: kendi anahtar karelerinden bir referans hareket üret ([Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator)) ve taklit ödülünü ona bağla.

### 10. Gerçek robota taşı

ONNX'lerimiz upstream arayüzünde (`obs` 1×101 → `continuous_actions`), yani [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) ile çalışması beklenir. Biz donanımda denemedik. Önce 6. deneyi yap: simde 3,23 N·m ile eğitilmiş politika gerçek servoda büyük olasılıkla yavaş kalır.
