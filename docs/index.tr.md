# open-duck-mac

![Open Duck Mini v2, MuJoCo: eğitilmemiş politika düşüyor, bizim 68,4 M adımdaki politikamız adım atıyor, hazır BEST_WALK_ONNX yürüyor](assets/hero.jpg)

**Open Duck Mini v2'ye bir MacBook'ta, sıfırdan, yalnız işlemciyle (CPU) yürümeyi öğretiyoruz. Sonra da onu bozup neden bozulduğunu ölçüyoruz.**

Bu site, kısa videolarımızın arkasındaki deneylerin tam hâli. Her deneyin betiği, logu ve ölçümü depoda. Hiçbir şeyi bize güvenerek kabul etmen gerekmiyor; kendi Mac'inde tekrar çalıştırabilirsin.

## Videolar

| Bölüm | Konu | YouTube |
|---|---|---|
| Ördek 1 | Viral ördeği almadım, kardeşini Mac'imde eğittim | [Shorts](https://youtube.com/shorts/gfCUTIHldt0) |
| Ördek 2 | Ördeğe "hızlı ol" dedim, yüzüstü sürünmeyi buldu | 28 Ekim 2026 |
| Ördek 3+ | Simülasyon yalan söyler (sim2real) | hazırlanıyor |

Kanal: [youtube.com/@getporti](https://www.youtube.com/@getporti)

## Bu sitede ne var?

<div class="grid cards" markdown>

- **[Kurulum](setup.md)**: Boş bir Mac'ten ilk eğitim koşusuna kadar her komut.
- **[Kavramlar](concepts.md)**: Pekiştirmeli öğrenme, PPO, ödül, simülasyon, domain randomization. Matematik bilmeden.
- **[1. Yürümeyi öğrenmek](part-1-walking.md)**: 3,6 saat, 72,4 M adım, 1.536 paralel ördek. Ne zaman ayağını kaldırdı?
- **[2. Ödül hilesi](part-2-reward-hacking.md)**: Yalnız ödülü değiştirdik; ördek sürünmeyi, donmayı, yatıp ayaklarını havaya dikmeyi buldu.
- **[3. Simülasyon yalan söyler](part-3-sim2real.md)**: Simdeki motor, gerçek servodan neredeyse iki kat güçlü. Ne oluyor? (devam ediyor)
- **[Sen ne yapabilirsin?](experiments.md)**: Zorluk ve süre tahminiyle, kendi başına deneyebileceğin fikirler.

</div>

## Kısaca sonuçlar

- Bir Apple M5 MacBook'ta (10 çekirdek, 6'sı kullanıldı) **3,57 saatte 72,4 milyon adım**. Mac'in GPU'su bu fiziği çalıştıramadığı için her şey işlemcide.
- Ördek **~11 dakikada** ayakta durmayı öğrendi, **~3 saat** yerinde saydı, **68,4 M** adımda ilk kez ayaklarını kaldırdı (5 sn'de 29,8 cm).
- Aynı politikadan devam edip yalnız ödülü bozunca: "hızlı ol" → **14,9 dk**'da yüzüstü sürünme, "enerji harcama" → **7,5 dk**'da donma, "ayaklarını kaldır" → **30 dk**'da yere yatıp ayakları havada.
- Motor gücü gerçek servo değerine (1,86 N·m) inince aynı politika **yarı mesafe** gidiyor; zemini kaydırmak ve ağırlık eklemek ise neredeyse hiç etkilemiyor.

!!! warning "Dürüst sınırlar"
    Her şey simülasyon; gerçek robot yok. Her deney tek seed, tek koşu. Resmî tarif NVIDIA GPU'da 150 M adım; biz %48'inde durduk. Ayrıntılar her bölümün sonunda.

## Teşekkür

Robot, model, ortam ve ödüller **Open Duck Mini** projesinin: [Antoine Pirrone](https://github.com/apirrone) ve katkıcılar ([Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini), [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)). Simülasyon [MuJoCo / MJX](https://github.com/google-deepmind/mujoco), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) ve [Brax](https://github.com/google/brax). Bunun üzerine bir şey kurarsan önce Open Duck Mini'yi an.
