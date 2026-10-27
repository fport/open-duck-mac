---
title: '3. Simülasyon yalan söyler (sim2real)'
---

:::note[Devam ediyor]
Bu bölümün ölçüm kısmı bitti, yeniden eğitim koşuları (27 Ekim 2026) sürüyor. Sonuçlar ve betikler gelince bu sayfa güncellenecek.
:::

## Soru

Simdeki ördek, gerçek ördekle aynı mı? Open Duck modelinde her eklem en fazla **±3,23 N·m** tork verebiliyor (`open_duck_mini_v2.xml`, `sts3215` sınıfı, `forcerange`). Gerçek STS3215 servosu için:

- bir topluluk projesi ([soyeon24/open-duck-mini-v2-build](https://github.com/soyeon24/open-duck-mini-v2-build)) servonun **~1,86 N·m**'de durduğunu yazıyor;
- satıcıların verdiği en yüksek (stall) tork **19,5 kg·cm ≈ 1,91 N·m** ([Core Electronics](https://core-electronics.com.au/feetech-sts3215-smart-servo.html), [RobotShop](https://www.robotshop.com/products/magnetic-encoding-servo-sts3215-74v-19kgcm)); hangi voltajda geçerli olduğu kaynaklar arasında tutarsız.

Yani simdeki motor gerçeğinden **%70–75 daha güçlü**. Resmî eğitim zemini, kütleyi, motor sertliğini ve gecikmeyi rastgeleleştiriyor ([Kavramlar](/open-duck-mac/concepts/#domain-randomization)) ama **tork sınırını değiştirmiyor**.

## Deney 1: yeniden eğitmeden, farklı dünyalar

Eğitilmiş policy'leri hiç değiştirmeden, yalnız dünyayı değiştirip ölçtük. Deterministik test, 5 sn, düz zemin, itme yok, 0,15 m/s.

| Dünya | Bizim 72,4 M | 9 gürültülü (bizim) | Hazır `BEST_WALK_ONNX` |
|---|---|---|---|
| Normal (3,23 N·m) | 28,8 cm | 0/9 düştü, ort. 22,1 cm | 58,5 cm · 8+8 adım |
| Kaygan zemin (sürtünme 0,5) | 24,9 cm | 0/9, ort. 20,6 cm | 57,3 cm |
| Kütle ×1,1 | 27,9 cm | 0/9, ort. 18,1 cm | 56,1 cm |
| Kütle ×1,2 | 24,7 cm | – | 54,3 cm |
| Tork 2,5 N·m | 25,3 cm | – | 52,3 cm |
| **Tork 1,86 N·m** | **13,9 cm (−%52)** | 0/9, ort. 9,2 cm | **40,2 cm (−%31)** · 8+5 adım |
| Tork 1,6 N·m | 9,8 cm | – | 27,3 cm · 8+0 adım |
| Tork 1,4 N·m | 5,8 cm | – | 19,5 cm · 7+0 adım |

![Aynı policy'nin dört motor gücünde 5 saniyelik ileri mesafesi: 3,23 N·m'de 28,8 cm, 2,5'te 25,3, gerçek servo değeri 1,86'da 13,9, 1,4'te 5,8 cm](../../assets/torque-distance.tr.png)

Ne görüyoruz?

- Eğitimde zaten değiştirilen şeyler (sürtünme, kütle) policy'yi neredeyse hiç etkilemiyor. Kütle ×1,2 eğitim aralığının (×0,9–1,1) dışında olduğu hâlde bile.
- Eğitimde hiç değiştirilmeyen tork sınırı mesafeyi yarıya indiriyor.
- Hiçbir ördek devrilmiyor; zayıf motor düşürmüyor, yavaşlatıyor. soyeon24 de aynı şeyi bildiriyor.
- Hazır policy 1,6 N·m'de sağ ayağıyla artık adım atamıyor (8+0).

Tekrar üretmek için [`scripts/sim2real/eval_world.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/sim2real/eval_world.py), `eval_checkpoint.py`'nin dünyayı ortam değişkenleriyle değiştiren kopyası:

```bash
FORCE=1.86 .venv/bin/python -I scripts/sim2real/eval_world.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/f186.mp4 --no_video        # "dx_m": 0.1393
FLOOR_FRIC=0.5 .venv/bin/python -I scripts/sim2real/eval_world.py ...            # kaygan zemin
MASS_SCALE=1.2 .venv/bin/python -I scripts/sim2real/eval_world.py ...            # tüm parçalar %20 ağır
FORCE_RAND=1.40,1.90 WSEED=3 .venv/bin/python -I scripts/sim2real/eval_world.py ...  # her eklem rastgele
```

Seçilen dünya çıktı JSON'unda `"world"` alanına yazılır.

## Deney 2: zayıf motorla yeniden eğitim (sürüyor)

Üç koşu, hepsi 68,4 M'den warm start, 10 M step; reward'lar, fizik ve resmî randomization aynı, tek fark tork sınırı:

| Koşu | Tork sınırı | Neden |
|---|---|---|
| `tork186` | her eklem sabit 1,86 N·m | simi gerçek servoya eşitle |
| `tork_rand` | her episode'da, her eklem için U(1,40, 1,90) N·m | domain randomization: tek bir değere ezber yapmasın |
| `kontrol323` | resmî 3,23 N·m | adil kıyas: düzelme yalnız "daha fazla eğitim"den mi geliyor? |

Hepsi sonra Deney 1'deki dünyalarda ölçülecek.
