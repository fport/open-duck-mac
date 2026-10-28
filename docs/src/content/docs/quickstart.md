---
title: "5 dakikada dene"
description: "Tek komutla kur, yürüyen ördeği izle, motorunu zayıflat, kendi ördeğini eğit."
---

Bu sayfa her şeyi tek bir komutla, `./duck` ile yapar. Uzun yolu ve her adımın ne yaptığını [Kurulum](/open-duck-mac/setup/) anlatıyor.

## Neye ihtiyacın var?

- Apple Silicon Mac (M1 ve sonrası). Yalnız macOS'ta denedik.
- `git` ve [`uv`](https://docs.astral.sh/uv/). Yoksa:

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv`'yi kurduktan sonra yeni bir terminal aç.

## 1. Kur (bir kez, 3–5 dk)

```bash
git clone https://github.com/fport/open-duck-mac.git
cd open-duck-mac
./duck setup
```

Ne yapıyor:

1. Ördeğin simülasyonunu ([Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)) bizim sonuçlarımızın commit'inde (`b9be205`) `external/` altına indirir.
2. Python 3.12 ortamını `.venv/` içine kurar (`requirements-lock.txt`, ilk sefer birkaç yüz MB).
3. Duman testi: simülasyon derleniyor ve adım atıyor mu?
4. İşlemciyi 4, 6 ve 8 parçaya bölüp hangisinin en hızlı eğittiğini ölçer, `.duck.json`'a yazar.

Bir şey ters giderse `./duck doctor` neyin eksik olduğunu ve nasıl düzeltileceğini yazar.

## 2. Yürüyen ördeği izle (birkaç saniye)

```bash
./duck try
```

Bizim 72,4 M adımda eğittiğimiz ördeği 5 saniye yürütür, videoyu açar ve ölçer:

```text
  mesafe (5 sn): 28.8 cm  ·  düşmedi
  ayaklar (sol / sağ): 8×, 11 mm / 8×, 11 mm  ·  sıralılık -0.34  →  YÜRÜYOR
```

"YÜRÜYOR" kararını [`gait.py`](/open-duck-mac/part-3-sim2real/#yürüyor-mu-kayıyor-mu-adım-sayacı-yanılıyordu) veriyor. En az 10 cm yol, iki ayakta da en az 4 salınım ve 5 mm kalkış gerekiyor. Sıralılık negatifse ayaklar sırayla hareket ediyor demektir.

## 3. Dünyayı boz (her biri bir saniyeden kısa)

Aynı ördek, motoru kutusundaki güce inmiş halde ([3. bölüm](/open-duck-mac/part-3-sim2real/)):

```bash
./duck eval duck1 --force 1.86 --no-video      # 13.9 cm: yarı yol
./duck eval catalog --force 1.86 --no-video    # 37.6 cm: o motorla eğitilmiş ördek
./duck eval forgot --force 1.86 --no-video     # 2.5 cm: aynı eğitim, 301 M'de unutmuş
```

Seçenekleri birleştirebilirsin; hepsini `./duck eval --list` gösterir:

| Seçenek | Ne değişir |
|---|---|
| `--force 1.86` | her motor en fazla 1,86 N·m (katalog değeri; simde 3,23) |
| `--force-rand 1.4 1.9` | her motor o aralıkta rastgele (`--world-seed` çekilişi seçer) |
| `--mass 1.2` | her parça %20 daha ağır |
| `--friction 0.5` | zemin sürtünmesi (normalde 0,6) |
| `--vx 0.3` | daha hızlı yürü komutu (normalde 0,15 m/s) |
| `--stochastic --seed 3` | eğitimdeki gibi gürültülü hareket |

Sınırı bul: ördek `--mass 3`'te 1 sn'de, `--force 0.4`'te 1,3 sn'de düşüyor; `--friction 0.05`'te kayıp yerinde sayıyor. Aradaki eşik nerede?

`--no-video`'yu çıkarırsan `out/eval/` altına video da çeker (birkaç saniye), `--open` ile açar.

## 4. Kendi ördeğini eğit

```bash
./duck train --preset quick --detach    # 10 M adım, bizim makinede ~30 dk
./duck watch                            # koşuları listeler
./duck watch quick-1028-1430            # her yeni checkpoint'i dener: yürüyor mu?
```

| Preset | Adım | Süre (bizim M5) | Ne beklemeli |
|---|---|---|---|
| `quick` | 10 M | ~30 dk | ayakta durmayı öğrenir, henüz yürümez |
| `full` | 72 M | ~3,6 sa | 1. bölümün tarifi; ilk adımlar 68 M civarı |
| `sim2real` | 150 M | ~7,5 sa | motor gücü 1,40–1,90 N·m arası rastgele; bizimki 88,8 M'de yürüdü |

- `--detach` eğitimi arka planda çalıştırır; terminali kapatabilirsin. Kapağı kapatmak Mac'i uyutur ve eğitimi durdurur.
- Eğitim düşük öncelikle (`nice`) çalışır; tarayıcı ve editör akıcı kalır, ama çekirdeklerin çoğu dolu olur.
- Aynı anda tek eğitim: ikincisi ikisini de yavaşlatır, `./duck train` buna izin vermez.
- `--seed 7` ile başka bir tohum dene; `--steps 20` adım sayısını (milyon) değiştirir.
- Eğitim ödülü yürüyüşü göstermez ([3. bölümde](/open-duck-mac/part-3-sim2real/#devam-301-me) ödül düşmeden yürüyüş kayboldu). Bu yüzden `./duck watch` her checkpoint'i ayrıca dener.

## Claude Code ile

Bu depoda bir [Claude Code](https://claude.com/claude-code) skill'i var: [`.claude/skills/open-duck-lab/SKILL.md`](https://github.com/fport/open-duck-mac/blob/main/.claude/skills/open-duck-lab/SKILL.md). Depoyu Claude Code'da açınca kendiliğinden yüklenir. Sonra "ördeği kur ve dene", "motoru zayıflatınca ne oluyor?", "kısa bir eğitim başlat" gibi isteklerle `./duck` komutlarını onun çalıştırmasını isteyebilirsin. Uzun bir eğitimi başlatmadan önce süreyi söyleyip onay ister.

## Sırada ne var?

- [Sen ne yapabilirsin?](/open-duck-mac/experiments/): kolaydan zora 10 deney.
- [Kavramlar](/open-duck-mac/concepts/): policy, reward, checkpoint, domain randomization.
- Takıldıysan: `./duck doctor`, sonra [Sorunlar ve SSS](/open-duck-mac/faq/).
