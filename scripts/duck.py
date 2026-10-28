"""duck: one entry point for open-duck-mac. Wraps the scripts in scripts/ so nobody has to type repo paths.

  ./duck setup                 install everything, run the smoke test, pick the fastest CPU layout
  ./duck doctor                check the install, print the fix for anything missing
  ./duck try                   video + numbers of our walking duck (a few seconds)
  ./duck eval <policy> [world] a policy in a changed world: --force 1.86, --mass 1.2, --friction 0.5 ...
  ./duck train --preset quick  train your own duck (quick 10 M ~30 min, full 72 M ~3.6 h, sim2real 150 M)
  ./duck watch [<run>]         test every new checkpoint of a run and say whether it walks

Messages are in Turkish when LANG starts with "tr" (or DUCK_LANG=tr), in English otherwise.
Only the Python standard library is used here, so it runs before the environment exists."""
import argparse, csv, json, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND = ROOT / "external" / "Open_Duck_Playground"
PLAYGROUND_URL = "https://github.com/apirrone/Open_Duck_Playground.git"
PIN = "b9be205ac64488c23504ca42e5ec790337adeec3"
VENV_PY = ROOT / ".venv" / "bin" / "python"
CONFIG = ROOT / ".duck.json"
RUNS, OUT = ROOT / "runs", ROOT / "out"
TR = (os.environ.get("DUCK_LANG") or os.environ.get("LC_ALL") or os.environ.get("LANG") or "").lower().startswith("tr")
TRAIN_RE = r"scripts/(train_mac|sim2real/train_sim2real|reward_hacking/train_hile)\.py"

# name: (onnx, English note, Turkish note)
POLICIES = {
    "duck1": ("results/2026-10-24/onnx/ckpt_072437760.onnx",
              "part 1, 72.4 M steps, trained with the sim motor (3.23 N·m)",
              "1. bölüm, 72,4 M step, simdeki motorla (3,23 N·m) eğitildi"),
    "duck1-first": ("results/2026-10-24/onnx/ckpt_068413440.onnx",
                    "part 1, 68.4 M steps, its very first steps",
                    "1. bölüm, 68,4 M step, ilk adımları"),
    "catalog": ("results/2026-10-27-sim2real/scratch_rand_cont/onnx/ckpt_137809920.onnx",
                "part 3, 137.8 M steps, trained with motor strength around the catalog value; walks at 1.86 N·m",
                "3. bölüm, 137,8 M step, katalog civarı motor gücüyle eğitildi; 1,86 N·m'de yürüyor"),
    "forgot": ("results/2026-10-27-sim2real/scratch_rand_cont/onnx/ckpt_301240320.onnx",
               "part 3, 301.2 M steps, the same run after it forgot how to walk",
               "3. bölüm, 301,2 M step, aynı koşu yürümeyi unuttuktan sonra"),
}
PRESETS = {  # script, steps, chunks, extra args, English note, Turkish note
    "quick": ("scripts/train_mac.py", 10_000_000, 4, [],
              "learns to stand, does not walk yet", "ayakta durmayı öğrenir, henüz yürümez"),
    "full": ("scripts/train_mac.py", 72_000_000, 19, [],
             "our part 1 recipe; first steps around 68 M", "1. bölümün tarifi; ilk adımlar 68 M civarı"),
    "sim2real": ("scripts/sim2real/train_sim2real.py", 150_000_000, 37, ["--force_rand", "1.40", "1.90"],
                 "motor strength U(1.40, 1.90) N·m; ours first walked at 88.8 M", "motor gücü U(1,40, 1,90) N·m; bizimki ilk kez 88,8 M'de yürüdü"),
}
OUR_STEPS_PER_S = 5555  # 72 M steps in 3.6 h on our M5 with 6 x 256

def t(en, tr): return tr if TR else en
def ok(msg): print(f"  \033[32m✓\033[0m {msg}", flush=True)
def warn(msg): print(f"  \033[33m!\033[0m {msg}", flush=True)
def bad(msg): print(f"  \033[31m✗\033[0m {msg}", flush=True)
def head(msg): print(f"\n\033[1m{msg}\033[0m", flush=True)
def die(msg): bad(msg); sys.exit(1)
def rel(p): p = Path(p).resolve(); return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

def sh(cmd, **kw):
    """Run a command, stream its output, return the exit code."""
    return subprocess.run([str(c) for c in cmd], **kw).returncode

def out_of(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr

def load_config():
    try: return json.loads(CONFIG.read_text())
    except (OSError, ValueError): return {}

def playground_commit():
    if not (PLAYGROUND / ".git").exists(): return None
    code, s = out_of(["git", "-C", PLAYGROUND, "rev-parse", "HEAD"])
    return s.strip() if code == 0 else None

def running_trainings():
    """'<pid> <run dir>' per running training; caffeinate's helper process carries the same command line, skip it."""
    code, s = out_of(["pgrep", "-fl", TRAIN_RE])
    out = []
    for l in (s.splitlines() if code == 0 else []):
        pid, _, cmd = l.strip().partition(" ")
        if not pid.isdigit() or cmd.startswith(("caffeinate", "nice")): continue
        m = re.search(r"\.py \S+ (\S+)", cmd)  # <script> <playground> <out_dir>
        out.append(f"{pid} {rel(m[1]) if m else cmd[:100]}")
    return out

def need_env():
    if not VENV_PY.exists() or not PLAYGROUND.exists():
        die(t("Not installed yet. Run: ./duck setup", "Henüz kurulmadı. Çalıştır: ./duck setup"))

def resolve_policy(name):
    if name in POLICIES: return ROOT / POLICIES[name][0]
    p = Path(name)
    if p.suffix == ".onnx" and p.exists(): return p.resolve()
    die(t(f"Unknown policy '{name}'. Use one of: {', '.join(POLICIES)} or a path to an .onnx file",
          f"Bilinmeyen policy '{name}'. Şunlardan biri: {', '.join(POLICIES)} ya da bir .onnx dosyası"))

def gait_of(js):
    sys.path.insert(0, str(ROOT / "scripts" / "sim2real"))
    from gait import gait, walks
    g = gait(js); return g, walks(g)

def rollout(onnx, mp4, world=None, video=True, small=True, extra=()):
    """eval_world.py on one policy; returns the result JSON (or exits with the error)."""
    env = dict(os.environ, **(world or {}))
    cmd = [VENV_PY, "-I", ROOT / "scripts/sim2real/eval_world.py", PLAYGROUND, onnx, mp4, *extra]
    cmd += ["--width", "540", "--height", "960"] if video and small else []
    cmd += [] if video else ["--no_video"]
    Path(mp4).parent.mkdir(parents=True, exist_ok=True)
    code, s = out_of(cmd, env=env)
    js = Path(mp4).with_suffix(".json")
    if code != 0 or not js.exists():
        print(s[-1500:]); die(t("The rollout failed (output above).", "Deneme başarısız (çıktı yukarıda)."))
    return json.loads(js.read_text())

def report(d, mp4=None, video=True):
    g, w = gait_of(d)
    fell = d.get("fell_t"); w = w and not fell
    fall = t(f"fell at {fell:.2f} s", f"{fell:.2f} sn'de düştü") if fell else t("did not fall", "düşmedi")
    print(f"  {t('distance', 'mesafe')} ({d.get('secs', 5):g} {t('s', 'sn')}): \033[1m{g['dx_cm']:.1f} cm\033[0m  ·  {fall}")
    feet = " / ".join(f"{f['cycles']}×, {f['lift_mm']:.0f} mm" for f in g["feet"])
    verdict = t("WALKS", "YÜRÜYOR") if w else t("does not walk", "yürümüyor")
    print(f"  {t('feet (L / R)', 'ayaklar (sol / sağ)')}: {feet}  ·  {t('alternation', 'sıralılık')} {g['lr_corr']:+.2f}  →  \033[1m{verdict}\033[0m")
    if mp4 and video: print(f"  {t('video', 'video')}: {rel(mp4)}")
    return g, w

# ---------------------------------------------------------------- setup / doctor
def cmd_setup(a):
    head(t("1/6 Tools", "1/6 Araçlar"))
    if sys.platform != "darwin" or platform.machine() != "arm64":
        warn(t("Tested only on Apple Silicon Macs; continuing anyway.", "Yalnız Apple Silicon Mac'te denendi; yine de devam."))
    else: ok(f"macOS {platform.mac_ver()[0]}, {platform.machine()}, {os.cpu_count()} {t('cores', 'çekirdek')}")
    if not shutil.which("git"): die(t("git missing: xcode-select --install", "git yok: xcode-select --install"))
    if not shutil.which("uv"):
        die(t("uv missing. Install it, open a new terminal, run ./duck setup again:\n      curl -LsSf https://astral.sh/uv/install.sh | sh",
              "uv yok. Kur, yeni bir terminal aç, ./duck setup'ı tekrar çalıştır:\n      curl -LsSf https://astral.sh/uv/install.sh | sh"))
    ok("git, uv")

    head(t("2/6 Open_Duck_Playground (the duck's simulation)", "2/6 Open_Duck_Playground (ördeğin simülasyonu)"))
    if not PLAYGROUND.exists():
        if sh(["git", "clone", "--quiet", PLAYGROUND_URL, PLAYGROUND]) != 0: die(t("git clone failed", "git clone başarısız"))
        if sh(["git", "-C", PLAYGROUND, "-c", "advice.detachedHead=false", "checkout", "--quiet", PIN]) != 0:
            die(t(f"could not check out {PIN[:7]}", f"{PIN[:7]} commit'ine geçilemedi"))
    head_ = playground_commit()
    if head_ == PIN: ok(f"{rel(PLAYGROUND)} @ {PIN[:7]}")
    else: warn(t(f"{rel(PLAYGROUND)} is at {str(head_)[:7]}, our results used {PIN[:7]}; leaving it as it is.",
                 f"{rel(PLAYGROUND)} {str(head_)[:7]}'de, bizim sonuçlar {PIN[:7]} ile; dokunmuyorum."))

    head(t("3/6 Python 3.12 environment (first time: a few hundred MB)", "3/6 Python 3.12 ortamı (ilk sefer birkaç yüz MB)"))
    if not VENV_PY.exists() and sh(["uv", "venv", "--quiet", "--python", "3.12", ROOT / ".venv"]) != 0:
        die(t("uv venv failed", "uv venv başarısız"))
    if sh(["uv", "pip", "install", "--quiet", "--python", VENV_PY, "-r", ROOT / "requirements-lock.txt"]) != 0:
        die(t("installing requirements-lock.txt failed", "requirements-lock.txt kurulamadı"))
    # mujoco_playground clones the whole mujoco_menagerie on first import; this robot does not need it
    code, s = out_of([VENV_PY, "-I", "-c", "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('mujoco_playground').origin))"])
    if code != 0: die(s)
    (Path(s.strip()) / "external_deps" / "mujoco_menagerie").mkdir(parents=True, exist_ok=True)
    ok(t("packages installed (requirements-lock.txt)", "paketler kuruldu (requirements-lock.txt)"))

    head(t("4/6 Smoke test: does the duck simulation build and step?", "4/6 Duman testi: ördek simülasyonu derleniyor ve adım atıyor mu?"))
    code, s = out_of([VENV_PY, "-I", ROOT / "scripts/smoke.py", PLAYGROUND, "--envs", "64"])
    m = re.search(r"steps/s=(\d+)", s)
    if code != 0 or not m: print(s[-1500:]); die(t("smoke test failed", "duman testi başarısız"))
    ok(t(f"64 ducks at once: {int(m[1]):,} steps/s", f"aynı anda 64 ördek: saniyede {int(m[1]):,} adım"))

    head(t("5/6 Fastest CPU layout for training", "5/6 Eğitim için en hızlı işlemci düzeni"))
    cfg = load_config()
    if a.no_bench and cfg.get("devices"): ok(t(f"kept {cfg['devices']} devices", f"{cfg['devices']} cihaz korundu"))
    else:
        n = os.cpu_count() or 4
        cands = [d for d in (a.devices or [4, 6, 8]) if d <= max(1, n - 2)] or [max(1, n - 2)]
        best = None
        for d in cands:
            code, s = out_of([VENV_PY, "-I", ROOT / "scripts/smoke_pmap.py", PLAYGROUND, "--devices", d, "--envs_per_dev", "256"])
            m = re.search(r"steps/s=(\d+)", s)
            if code != 0 or not m: warn(t(f"{d} devices failed", f"{d} cihaz başarısız")); continue
            sps = int(m[1]); print(f"    {d} × 256: {sps:,} {t('steps/s', 'adım/sn')}", flush=True)
            if not best or sps > best[1]: best = (d, sps)
        if not best: die(t("no layout worked", "hiçbir düzen çalışmadı"))
        cfg.update(devices=best[0], envs_per_dev=256, smoke_steps_per_s=best[1])
        ok(t(f"using {best[0]} devices × 256 ducks", f"{best[0]} cihaz × 256 ördek kullanılacak"))

    head(t("6/6 Saving", "6/6 Kaydediliyor"))
    cfg.update(playground_commit=playground_commit(), setup_at=time.strftime("%Y-%m-%d %H:%M"))
    CONFIG.write_text(json.dumps(cfg, indent=1) + "\n"); ok(rel(CONFIG))
    print("\n" + t("Ready. Next: ./duck try", "Hazır. Sıradaki: ./duck try"))

def cmd_doctor(a):
    probs = 0
    head(t("System", "Sistem"))
    if sys.platform == "darwin" and platform.machine() == "arm64":
        ok(f"macOS {platform.mac_ver()[0]}, Apple Silicon, {os.cpu_count()} {t('cores', 'çekirdek')}")
    else: warn(t("not an Apple Silicon Mac: untested", "Apple Silicon Mac değil: denenmedi"))
    free = shutil.disk_usage(ROOT).free / 1e9
    (ok if free > 5 else warn)(t(f"{free:.0f} GB free disk", f"{free:.0f} GB boş disk"))
    for tool, need, hint in (("git", True, "xcode-select --install"), ("uv", True, "curl -LsSf https://astral.sh/uv/install.sh | sh"),
                             ("ffmpeg", False, "brew install ffmpeg")):
        if shutil.which(tool): ok(tool)
        elif need: bad(f"{tool} {t('missing', 'yok')}: {hint}"); probs += 1
        else: warn(f"{tool} {t('missing (optional, only for the 3×3 video grid)', 'yok (isteğe bağlı, yalnız 3×3 video ızgarası için)')}: {hint}")
    head(t("Install", "Kurulum"))
    c = playground_commit()
    if c is None: bad(t("Open_Duck_Playground missing → ./duck setup", "Open_Duck_Playground yok → ./duck setup")); probs += 1
    elif c == PIN: ok(f"Open_Duck_Playground @ {PIN[:7]}")
    else: warn(t(f"Open_Duck_Playground at {c[:7]}, results used {PIN[:7]}", f"Open_Duck_Playground {c[:7]}'de, sonuçlar {PIN[:7]} ile"))
    if not VENV_PY.exists(): bad(t(".venv missing → ./duck setup", ".venv yok → ./duck setup")); probs += 1
    else:
        code, s = out_of([VENV_PY, "-I", "-c", "import sys, jax, mujoco, mujoco_playground, brax, onnxruntime; "
                          "print(sys.version.split()[0], jax.__version__, mujoco.__version__, brax.__version__)"])
        if code == 0:
            py, jx, mj, bx = s.split()[-4:]; ok(f"Python {py} · jax {jx} · mujoco {mj} · brax {bx}")
            if not py.startswith("3.12"): warn(t("we used Python 3.12", "biz Python 3.12 kullandık"))
        else: bad(t("packages do not import → ./duck setup", "paketler import edilemiyor → ./duck setup")); print(s[-800:]); probs += 1
    cfg = load_config()
    if cfg.get("devices"): ok(t(f"training layout: {cfg['devices']} devices × {cfg.get('envs_per_dev', 256)}",
                                 f"eğitim düzeni: {cfg['devices']} cihaz × {cfg.get('envs_per_dev', 256)}"))
    else: warn(t("no .duck.json yet (./duck setup measures the fastest layout)", ".duck.json yok (./duck setup en hızlı düzeni ölçer)"))
    head(t("Now", "Şu an"))
    tr = running_trainings()
    if tr: warn(t(f"{len(tr)} training(s) running; they use most of the CPU:", f"{len(tr)} eğitim çalışıyor; işlemcinin çoğunu kullanıyor:")); [print("    pid " + l) for l in tr]
    else: ok(t("no training running", "çalışan eğitim yok"))
    print("\n" + (t("All good. Try: ./duck try", "Her şey yolunda. Dene: ./duck try") if not probs
                  else t(f"{probs} problem(s). Fix them, or see docs → Troubleshooting & FAQ.", f"{probs} sorun var. Düzelt ya da dokümanda Sorunlar ve SSS'ye bak.")))
    sys.exit(1 if probs else 0)

# ---------------------------------------------------------------- try / eval
def cmd_try(a):
    need_env()
    onnx = ROOT / POLICIES["duck1"][0]; mp4 = OUT / "try" / "duck1.mp4"
    head(t("Our duck (part 1, 72.4 M training steps), 5 s, clean floor, command 0.15 m/s",
           "Bizim ördek (1. bölüm, 72,4 M eğitim adımı), 5 sn, temiz zemin, komut 0,15 m/s"))
    print("  " + t("simulating and rendering…", "simüle ediliyor ve çekiliyor…"), flush=True)
    t0 = time.time(); d = rollout(onnx, mp4); print(f"  ({time.time() - t0:.0f} s)")
    report(d, mp4)
    if not a.no_open and sys.platform == "darwin": subprocess.run(["open", mp4])
    print("\n" + t("Next, change the world and see what breaks (each under a second without video):",
                   "Sırada: dünyayı değiştir, neyin bozulduğuna bak (videosuz her biri bir saniyeden kısa):"))
    for c, en, tr_ in (("./duck eval duck1 --force 1.86 --no-video", "motor at the catalog value: half the distance", "motor katalog değerinde: yarı yol"),
                       ("./duck eval catalog --force 1.86 --no-video", "a duck trained for that motor", "o motor için eğitilmiş ördek"),
                       ("./duck eval duck1 --mass 1.2 --no-video", "20 % heavier", "%20 daha ağır"),
                       ("./duck eval --list", "all policies and world options", "bütün policy'ler ve dünya seçenekleri")):
        print(f"  {c:<46} {t(en, tr_)}")

def world_of(a):
    w, tag = {}, []
    if a.force is not None: w["FORCE"] = str(a.force); tag.append(f"force{a.force:g}")
    if a.force_rand: w["FORCE_RAND"] = f"{a.force_rand[0]},{a.force_rand[1]}"; w["WSEED"] = str(a.world_seed); tag.append(f"rand{a.force_rand[0]:g}-{a.force_rand[1]:g}s{a.world_seed}")
    if a.friction is not None: w["FLOOR_FRIC"] = str(a.friction); tag.append(f"fric{a.friction:g}")
    if a.mass is not None: w["MASS_SCALE"] = str(a.mass); tag.append(f"mass{a.mass:g}")
    return w, "_".join(tag) or "sim"

def cmd_eval(a):
    if a.list or not a.policy:
        head(t("Policies", "Policy'ler"))
        for k, (p, en, tr_) in POLICIES.items(): print(f"  {k:<12} {t(en, tr_)}")
        print("  " + t("…or any .onnx path, e.g. runs/<run>/ckpt/ckpt_010000000.onnx", "…ya da herhangi bir .onnx yolu, ör. runs/<koşu>/ckpt/ckpt_010000000.onnx"))
        head(t("World (combine freely)", "Dünya (istediğin gibi birleştir)"))
        for o, en, tr_ in (("--force 1.86", "every motor limited to 1.86 N·m (catalog); sim default 3.23", "her motor 1,86 N·m'ye sınırlı (katalog); sim varsayılanı 3,23"),
                           ("--force-rand 1.4 1.9", "each motor random in that range (--world-seed picks the draw)", "her motor o aralıkta rastgele (--world-seed çekilişi seçer)"),
                           ("--mass 1.2", "every part 20 % heavier", "her parça %20 ağır"),
                           ("--friction 0.5", "floor friction (default 0.6)", "zemin sürtünmesi (varsayılan 0,6)"),
                           ("--vx 0.3", "faster forward command (default 0.15 m/s)", "daha hızlı ileri komutu (varsayılan 0,15 m/s)"),
                           ("--stochastic --seed 3", "noisy actions, as during training", "eğitimdeki gibi gürültülü hareket")):
            print(f"  {o:<24} {t(en, tr_)}")
        print("\n  " + t("Example: ./duck eval duck1 --force 1.86", "Örnek: ./duck eval duck1 --force 1.86"))
        return
    need_env()
    onnx = resolve_policy(a.policy); w, tag = world_of(a)
    name = a.policy if a.policy in POLICIES else Path(a.policy).stem
    if a.stochastic: tag += f"_noise{a.seed}"
    if a.vx != 0.15: tag += f"_vx{a.vx:g}"
    mp4 = OUT / "eval" / f"{name}__{tag}.mp4"
    head(f"{name} · {tag}")
    extra = ["--vx", str(a.vx), "--secs", str(a.secs)] + (["--stochastic", "--seed", str(a.seed)] if a.stochastic else [])
    d = rollout(onnx, mp4, w, video=not a.no_video, extra=extra)
    report(d, mp4, video=not a.no_video)
    if not a.no_video and a.open and sys.platform == "darwin": subprocess.run(["open", mp4])

# ---------------------------------------------------------------- train / watch
def cmd_train(a):
    need_env()
    script, steps, chunks, extra, en, tr_ = PRESETS[a.preset]
    if a.steps is not None:
        if a.steps <= 0: die(t("--steps must be > 0 (millions of steps)", "--steps 0'dan büyük olmalı (milyon adım)"))
        steps = int(a.steps * 1e6); chunks = max(1, round(steps / 4e6))
    if a.chunks is not None:
        if a.chunks < 1: die(t("--chunks must be >= 1", "--chunks en az 1 olmalı"))
        chunks = a.chunks
    cfg = load_config(); devices = a.devices or cfg.get("devices", 6); epd = cfg.get("envs_per_dev", 256)
    name = a.name or f"{a.preset}-{time.strftime('%m%d-%H%M')}"
    run = RUNS / name
    if run.exists() and any(run.iterdir()): die(t(f"{rel(run)} already exists; pick another --name", f"{rel(run)} zaten var; başka bir --name seç"))
    busy = running_trainings()
    if busy and not a.force: die(t("A training is already running (./duck doctor shows it). Two at once halve each other's speed; add --force to start anyway.",
                                   "Zaten bir eğitim çalışıyor (./duck doctor gösterir). İkisi aynı anda birbirini yarı hıza düşürür; yine de başlatmak için --force ekle."))
    sps = cfg.get("smoke_steps_per_s"); hours = steps / (OUR_STEPS_PER_S * (sps / 4621 if sps else 1)) / 3600
    head(f"{a.preset}: {steps / 1e6:g} M {t('steps', 'adım')}, {chunks} {t('checkpoints', 'checkpoint')}, {devices} × {epd} · {t(en, tr_)}")
    print(f"  {t('estimated', 'tahmini süre')}: ~{hours:.1f} {t('h', 'sa')} · {t('output', 'çıktı')}: {rel(run)}/")
    cmd = ["nice", "-n", "10", "caffeinate", "-i", VENV_PY, "-I", ROOT / script, PLAYGROUND, run,
           "--devices", devices, "--envs_per_dev", epd, "--num_timesteps", steps, "--num_evals", chunks, "--seed", a.seed, *extra]
    run.mkdir(parents=True, exist_ok=True); log = run / "train.log"
    if a.detach:
        with open(log, "w") as f: p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (run / "pid").write_text(str(p.pid))
        print("\n" + t(f"Running in the background (pid {p.pid}). Closing the terminal is fine; closing the lid sleeps the Mac and pauses it.",
                       f"Arka planda çalışıyor (pid {p.pid}). Terminali kapatabilirsin; kapağı kapatmak Mac'i uyutur ve eğitimi durdurur."))
        print(f"  ./duck watch {name}          {t('test each new checkpoint', 'her yeni checkpoint’i dene')}")
        print(f"  tail -f {rel(log)}   {t('raw log', 'ham log')}")
        print(f"  kill {p.pid}                    {t('stop', 'durdur')}")
        return
    print("  " + t(f"Ctrl+C stops it. For long runs use --detach. Log: {rel(log)}", f"Ctrl+C durdurur. Uzun koşular için --detach. Log: {rel(log)}"), flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout: sys.stdout.write(line); f.write(line); f.flush()
        except KeyboardInterrupt: p.terminate(); print("\n" + t("stopped", "durduruldu")); sys.exit(130)
    sys.exit(p.wait())

def cmd_watch(a):
    need_env()
    if not a.run:
        runs = sorted((r for r in RUNS.glob("*") if (r / "ckpt").exists()), key=lambda r: r.stat().st_mtime) if RUNS.exists() else []
        if not runs: die(t("No runs yet. Start one: ./duck train --preset quick", "Henüz koşu yok. Başlat: ./duck train --preset quick"))
        head(t("Runs", "Koşular"))
        for r in runs: print(f"  {r.name:<28} {len(list((r / 'ckpt').glob('*.onnx')))} checkpoint")
        print("\n  ./duck watch <" + t("run", "koşu") + ">"); return
    run = RUNS / a.run if (RUNS / a.run).exists() else Path(a.run)
    if not (run / "ckpt").exists(): die(t(f"{a.run}: no ckpt/ folder", f"{a.run}: ckpt/ klasörü yok"))
    force = a.force
    if force is None:
        try: force = "1.86" if json.loads((run / "run_config.json").read_text()).get("force_limit") else "sim"
        except (OSError, ValueError): force = "sim"
    head(t(f"{run.name}: every checkpoint, 5 s, motor {'3.23 (sim)' if force == 'sim' else force + ' N·m'}",
           f"{run.name}: her checkpoint, 5 sn, motor {'3,23 (sim)' if force == 'sim' else force + ' N·m'}"))
    if not a.once: print("  " + t("checks every 2 min until the training log says DONE; Ctrl+C stops watching (not the training)",
                                  "eğitim logu DONE diyene kadar 2 dk'da bir bakar; Ctrl+C izlemeyi durdurur (eğitimi değil)"), flush=True)
    csv_path = run / ("watch.csv" if force == "sim" else f"watch_f{force.replace('.', '')}.csv")
    cmd = [sys.executable, "-I", ROOT / "scripts/sim2real/watch_ckpts.py", run / "ckpt", run / "train.log", csv_path, "--force", force]
    if a.once: cmd.append("--once")
    def row(tag, dx, walks): print(f"  {int(tag[5:]) / 1e6:>10.1f} M  {dx:>8} cm   {'✓' if walks else '·'}", flush=True)
    print(f"\n  {'checkpoint':>12}  {t('distance', 'mesafe'):>11}   {t('walks?', 'yürüyor?')}")
    for r in (csv.DictReader(open(csv_path)) if csv_path.exists() else []): row(r["tag"], r["dx_cm"], r["walks"] == "1")
    p = subprocess.Popen([str(c) for c in cmd], env=dict(os.environ, PY=str(VENV_PY), PLAYGROUND=str(PLAYGROUND)),
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        for line in p.stdout:  # watch_ckpts.py prints "<tag> <gait json>" per checkpoint
            tag, _, js = line.strip().partition(" ")
            if not tag.startswith("ckpt_"): continue
            g = json.loads(js) if js and js != "null" else None
            if g is None: print(f"  {tag}: {t('evaluation failed', 'değerlendirme başarısız')}"); continue
            sys.path.insert(0, str(ROOT / "scripts" / "sim2real")); from gait import walks
            row(tag, f"{g['dx_cm']:.1f}", walks(g))
        p.wait()
    except KeyboardInterrupt: p.terminate(); print()
    print("\n  " + t(f"table: {rel(csv_path)} · reward curve: .venv/bin/python scripts/make_log.py {rel(run)}",
                     f"tablo: {rel(csv_path)} · reward eğrisi: .venv/bin/python scripts/make_log.py {rel(run)}"))

def main():
    ap = argparse.ArgumentParser(prog="duck", description=t("open-duck-mac: train and test the Open Duck Mini v2 on a Mac CPU",
                                                            "open-duck-mac: Open Duck Mini v2'yi Mac işlemcisinde eğit ve dene"))
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    p = sub.add_parser("setup", help=t("install everything and pick the fastest CPU layout (~3–5 min)", "her şeyi kur, en hızlı işlemci düzenini seç (~3–5 dk)"))
    p.add_argument("--no-bench", action="store_true", help=t("keep the saved CPU layout", "kayıtlı işlemci düzenini koru"))
    p.add_argument("--devices", type=int, nargs="+", help=t("CPU layouts to measure (default 4 6 8)", "ölçülecek düzenler (varsayılan 4 6 8)"))
    sub.add_parser("doctor", help=t("check the install and say how to fix it", "kurulumu kontrol et, nasıl düzelteceğini söyle"))
    p = sub.add_parser("try", help=t("video + numbers of our walking duck (a few seconds)", "yürüyen ördeğimizin videosu ve ölçümü (birkaç saniye)"))
    p.add_argument("--no-open", action="store_true", help=t("do not open the video", "videoyu açma"))
    p = sub.add_parser("eval", help=t("test a policy in a changed world (--list for options)", "bir policy'yi değişmiş dünyada dene (--list seçenekleri gösterir)"))
    p.add_argument("policy", nargs="?", help=f"{', '.join(POLICIES)} {t('or a path to an .onnx', 'ya da bir .onnx yolu')}")
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", type=float, help="N·m")
    p.add_argument("--force-rand", type=float, nargs=2, metavar=("LO", "HI"))
    p.add_argument("--world-seed", type=int, default=0)
    p.add_argument("--mass", type=float); p.add_argument("--friction", type=float)
    p.add_argument("--vx", type=float, default=0.15); p.add_argument("--secs", type=float, default=5.0)
    p.add_argument("--stochastic", action="store_true"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true"); p.add_argument("--open", action="store_true", help=t("open the video", "videoyu aç"))
    p = sub.add_parser("train", help=t("train your own duck (--preset quick | full | sim2real)", "kendi ördeğini eğit (--preset quick | full | sim2real)"))
    p.add_argument("--preset", choices=list(PRESETS), default="quick")
    p.add_argument("--name"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=float, help=t("millions of steps (overrides the preset)", "milyon adım (preset'i ezer)"))
    p.add_argument("--chunks", type=int, help=t("number of checkpoints", "checkpoint sayısı"))
    p.add_argument("--devices", type=int, help=t("CPU devices (default: from ./duck setup)", "işlemci cihazı (varsayılan: ./duck setup'tan)"))
    p.add_argument("--detach", action="store_true", help=t("run in the background", "arka planda çalıştır"))
    p.add_argument("--force", action="store_true", help=t("start even if another training runs", "başka eğitim çalışsa da başlat"))
    p = sub.add_parser("watch", help=t("test each checkpoint of a run: does it walk?", "bir koşunun her checkpoint'ini dene: yürüyor mu?"))
    p.add_argument("run", nargs="?"); p.add_argument("--once", action="store_true", help=t("one pass, then exit", "bir tur bak, çık"))
    p.add_argument("--force", help=t("motor limit for the test: 'sim' (3.23) or N·m; default from the run", "test motor sınırı: 'sim' (3,23) ya da N·m; varsayılan koşudan"))
    a = ap.parse_args(["-h"] if sys.argv[1:2] == ["help"] else None)
    if not a.cmd: ap.print_help(); return
    {"setup": cmd_setup, "doctor": cmd_doctor, "try": cmd_try, "eval": cmd_eval, "train": cmd_train, "watch": cmd_watch}[a.cmd](a)

if __name__ == "__main__":
    main()
