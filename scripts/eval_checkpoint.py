"""Roll out an ONNX policy for Open Duck Mini v2 in plain MuJoCo (CPU) and measure it; optionally render a video.

Works for checkpoints from train_mac.py and for pretrained policies (e.g. Open_Duck_Mini's BEST_WALK_ONNX.onnx).
The deterministic test used for every number in results/: clean flat floor, no pushes, constant forward
command vx = 0.15 m/s, 5 s, policy mean action (tanh(loc)). With --stochastic, actions are sampled with
the policy's exploration noise, a = tanh(loc + std * eps), as during training (eval_grid.py runs 9 seeds).

Physics and control loop: Open_Duck_Playground's MjInfer (playground/open_duck_mini_v2/mujoco_infer.py)
and its MJCF, unchanged; only the scene around the robot (lights, sky, floor texture) is ours, with the
same floor geom/friction and home keyframe as scene_flat_terrain.xml. Training itself runs in MJX, so
small numerical differences to this C-MuJoCo rollout are possible.

One render-only addition: in the training model only the feet collide with the floor (a fall ends the
episode). So that a falling duck does not sink through the floor on video, once the trunk starts to tip
(up-axis z < 0.6 or trunk height < 0.11 m) the robot's meshes are made floor-collidable. While the duck
is upright the physics is identical to training.

Writes <out>.json (dist_m, dx_m, fell_t, min/final trunk height, foot_steps, foot_z_range, per-frame
trace) and, unless --no_video, <out>.mp4 + <out>_first.png + <out>_last.png.
"fell" = trunk tipped > 60 deg (up-axis z < 0.5) or trunk below 0.09 m (stricter than training's > 90 deg).
"foot step" = a foot site rises > 1 cm above its lowest height so far and comes back down (< 0.4 cm).

  python -I scripts/eval_checkpoint.py ../Open_Duck_Playground runs/run1/ckpt/ckpt_072437760.onnx \\
      runs/run1/sim/ckpt_072437760.mp4 [--secs 5] [--vx 0.15] [--width 1080] [--height 1920]
      [--stochastic --seed 3] [--no_video]
"""
import argparse, json, os, pathlib, sys, tempfile, types

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", help="path to an Open_Duck_Playground checkout")
ap.add_argument("onnx", help="policy .onnx (input 'obs', output 'continuous_actions')")
ap.add_argument("out", help="output path ending in .mp4; the .json (and PNGs) are written next to it")
ap.add_argument("--secs", type=float, default=5.0)
ap.add_argument("--vx", type=float, default=0.15, help="forward velocity command, m/s (default 0.15)")
ap.add_argument("--width", type=int, default=1080)
ap.add_argument("--height", type=int, default=1920)
ap.add_argument("--stochastic", action="store_true", help="sample actions with exploration noise (needs the "
                "loc_out/scale outputs that onnx_export.py writes)")
ap.add_argument("--seed", type=int, default=0, help="noise seed for --stochastic")
ap.add_argument("--azimuth", type=float, nargs=2, default=[18.0, 44.0], metavar=("START", "END"),
                help="camera azimuth sweep in degrees (default 18 44)")
ap.add_argument("--no_video", action="store_true", help="metrics only: no renderer, no mp4/png (same physics)")
a = ap.parse_args()

repo, onnx_path, out = os.path.abspath(a.repo), os.path.abspath(a.onnx), a.out
if not out.endswith(".mp4"):
    sys.exit("out must end in .mp4")
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
secs, vx, W, H = a.secs, a.vx, a.width, a.height
sys.path.append(repo)
import numpy as np, mujoco

# Stub the two playground modules that pull in jax/MJX: MjInfer only needs get_assets() and the filter class.
root = pathlib.Path(repo) / "playground/open_duck_mini_v2"
def get_assets():
    assets = {}
    for p in (root / "xmls").glob("*.xml"): assets[p.name] = p.read_bytes()
    for p in (root / "xmls/assets").glob("*"): assets[p.name] = p.read_bytes()
    return assets
base = types.ModuleType("playground.open_duck_mini_v2.base"); base.get_assets = get_assets
sys.modules["playground.open_duck_mini_v2.base"] = base
utils = types.ModuleType("playground.common.utils")
class LowPassActionFilter:
    def __init__(self, *args, **kwargs): pass
utils.LowPassActionFilter = LowPassActionFilter
sys.modules["playground.common.utils"] = utils
import playground.open_duck_mini_v2 as pkg; pkg.base = base
from playground.open_duck_mini_v2.mujoco_infer import MjInfer

# Same physics as scene_flat_terrain.xml (floor geom, friction, home key); only visuals differ.
SCENE = f"""
<mujoco model="eval_scene">
  <include file="open_duck_mini_v2.xml"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="38"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight diffuse="0.10 0.10 0.12" ambient="0.10 0.10 0.13" specular="0 0 0"/>
    <rgba haze="0.02 0.025 0.05 1"/>
    <map znear="0.01" zfar="40" haze="0.6"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.035 0.04 0.08" rgb2="0.004 0.005 0.012" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.075 0.08 0.10" rgb2="0.07 0.075 0.095"
             markrgb="0.20 0.21 0.25" width="400" height="400"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="6 6" reflectance="0.22" specular="0.4" shininess="0.6"/>
  </asset>
  <worldbody>
    <light name="key" mode="trackcom" pos="-0.35 -0.75 0.85" dir="0.35 0.75 -0.85" diffuse="1.3 1.0 0.78" specular="0.5 0.35 0.25"
           castshadow="true" cutoff="40" exponent="6" attenuation="1 0 0"/>
    <light name="fill" mode="trackcom" pos="-0.8 0.2 0.5" dir="0.8 -0.2 -0.5" diffuse="0.20 0.23 0.32" specular="0 0 0" castshadow="false"/>
    <light name="rim" mode="trackcom" pos="0.6 0.45 0.7" dir="-0.6 -0.45 -0.7" diffuse="0.35 0.38 0.48" specular="0.2 0.2 0.25" castshadow="false"/>
    <body name="floor">
      <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0 0.15 1 0 0 0 0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.003 -0.065 0.635 1.379 -0.796"
         ctrl="0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.003 -0.065 0.635 1.379 -0.796"/>
  </keyframe>
</mujoco>
"""
tmp = pathlib.Path(tempfile.mkdtemp()) / "eval_scene.xml"
tmp.write_text(SCENE)

m = MjInfer(str(tmp), str(root / "data/polynomial_coefficients.pkl"), onnx_path, False)
m.commands[0] = vx
if a.stochastic:
    import onnxruntime as ort
    class StochasticPolicy:
        def __init__(self, path, seed):
            self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.rng = np.random.default_rng(seed)
        def infer(self, obs):
            _, loc, sc = self.s.run(None, {"obs": np.asarray([obs], dtype=np.float32)})
            return np.tanh(loc[0] + sc[0] * self.rng.standard_normal(loc.shape[1])).astype(np.float32)
    m.policy = StochasticPolicy(onnx_path, a.seed)
model, data = m.model, m.data
fps = 30
head_id = model.body("head_assembly").id
trunk_id = model.body("trunk_assembly").id

r = w = None
if not a.no_video:
    import imageio
    r = mujoco.Renderer(model, height=H, width=W)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=None, bitrate="60M", macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "14"])
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
LOOK_DZ = -0.035  # look slightly below the trunk
AZ0, AZ1 = a.azimuth
ease = lambda u: u * u * (3 - 2 * u)
def cam_at(t):
    u = ease(min(1.0, max(0.0, t / secs)))
    cam.distance = 2.0 - 0.45 * u
    cam.azimuth = AZ0 + (AZ1 - AZ0) * u
    cam.elevation = -8 - 5 * u
    p = data.xpos[trunk_id]
    cam.lookat[:] = [p[0], p[1], p[2] + LOOK_DZ]

def project(p):
    """World point -> output pixel coordinates (for overlays)."""
    c = r.scene.camera[0]
    pos = np.array(c.pos); fw = np.array(c.forward); upv = np.array(c.up)
    right = np.cross(fw, upv)
    rel = np.asarray(p) - pos
    x, y, z = rel @ right, rel @ upv, rel @ fw
    n = c.frustum_near
    hw = c.frustum_width if c.frustum_width > 0 else (c.frustum_top - c.frustum_bottom) / 2 * W / H
    sx = ((x * n / z) - (c.frustum_center - hw)) / (2 * hw) * W
    sy = (c.frustum_top - y * n / z) / (c.frustum_top - c.frustum_bottom) * H
    return [round(float(sx), 1), round(float(sy), 1)]

steps_per_frame = int(round(1 / fps / m.sim_dt))
n_frames = int(secs * fps)
counter = 0
start_xy = data.qpos[:2].copy()
mujoco.mj_forward(model, data)
foot_sites = [model.site(n).id for n in ("left_foot", "right_foot")]
FOOT_Z0 = [float(data.site_xpos[s][2]) for s in foot_sites]
frames = []
FALL_Z = 0.09  # home trunk z = 0.15
min_z, fell_t = 1e9, None
robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and model.geom_contype[g] == 0
               and model.geom_conaffinity[g] == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
collide_on_t = None
foot_up = [False, False]; foot_steps = [0, 0]; foot_zr = [[9, -9], [9, -9]]
for f in range(n_frames):
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data); counter += 1
        if counter % m.decimation == 0:  # 50 Hz policy, same order of operations as MjInfer.run()
            m.imitation_i = (m.imitation_i + m.phase_frequency_factor) % m.PRM.nb_steps_in_period
            ph = m.imitation_i / m.PRM.nb_steps_in_period * 2 * np.pi
            m.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
            obs = m.get_obs(data, m.commands)
            action = m.policy.infer(obs)
            m.last_last_last_action = m.last_last_action.copy(); m.last_last_action = m.last_action.copy(); m.last_action = action.copy()
            m.motor_targets = m.default_actuator + action * m.action_scale
            lim = m.max_motor_velocity * (m.sim_dt * m.decimation)
            m.motor_targets = np.clip(m.motor_targets, m.prev_motor_targets - lim, m.prev_motor_targets + lim)
            m.prev_motor_targets = m.motor_targets.copy()
            data.ctrl = m.motor_targets.copy()
    fr = {"t": round(f / fps, 4), "dist": round(float(np.linalg.norm(data.qpos[:2] - start_xy)), 4),
          "dx": round(float(data.qpos[0] - start_xy[0]), 4), "fz": [round(float(data.site_xpos[s][2]), 4) for s in foot_sites]}
    if r is not None:
        cam_at(f / fps)
        r.update_scene(data, cam)
        fr.update(head=project(data.xpos[head_id]), trunk=project(data.xpos[trunk_id]))
        img = r.render(); w.append_data(img)
        if f == 0: imageio.imwrite(out.replace(".mp4", "_first.png"), img)
        if f == n_frames - 1: imageio.imwrite(out.replace(".mp4", "_last.png"), img)
    frames.append(fr)
    tz = float(data.qpos[2]); min_z = min(min_z, tz)
    upz = float(data.xmat[trunk_id].reshape(3, 3)[2, 2])
    if collide_on_t is None and (upz < 0.6 or tz < 0.11):
        model.geom_conaffinity[robot_geoms] = 1; collide_on_t = round(f / fps, 3)
        if hasattr(model, "body_conaffinity"):  # the broadphase body filter must be updated too
            for g in robot_geoms: model.body_conaffinity[model.geom_bodyid[g]] |= 1
    for k, sid in enumerate(foot_sites):
        z = float(data.site_xpos[sid][2])
        if f < 10: continue  # first 1/3 s: the duck settles onto the floor from the home pose
        foot_zr[k] = [min(foot_zr[k][0], z), max(foot_zr[k][1], z)]
        lowest = foot_zr[k][0]
        if not foot_up[k] and z > lowest + 0.01: foot_up[k] = True
        elif foot_up[k] and z < lowest + 0.004: foot_up[k] = False; foot_steps[k] += 1
    if fell_t is None and (upz < 0.5 or tz < FALL_Z): fell_t = round(f / fps, 3)
if w: w.close()
dist = float(np.linalg.norm(data.qpos[:2] - start_xy))
summary = {"stochastic": a.stochastic, "seed": a.seed, "fps": fps, "width": W, "height": H, "vx": vx, "secs": secs,
           "onnx": os.path.basename(onnx_path), "dist_m": round(dist, 4), "dx_m": round(float(data.qpos[0] - start_xy[0]), 4),
           "final_trunk_z": round(float(data.qpos[2]), 4), "min_trunk_z": round(min_z, 4), "fell_t": fell_t,
           "floor_collision_on_t": collide_on_t, "foot_steps": foot_steps, "foot_z_range": foot_zr, "foot_z0": FOOT_Z0}
with open(out.replace(".mp4", ".json"), "w") as fh:
    json.dump(summary | {"frames": frames}, fh)
print(json.dumps(summary))
