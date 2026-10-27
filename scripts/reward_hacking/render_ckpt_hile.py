"""[reward hacking version: copy of our video renderer (a restyled ../eval_checkpoint.py). Differences: (1) <repo> = Playground_hile (make_hile_repo.py),
trunk/head/leg box <-> floor <pair>s are added to the scene as in training; (2) post-fall "render-only" mesh
collision OFF (so physics matches training); (3) extra measurements in the json: torque², fraction of frames with both feet in the air, max
tilt, time flipped, forward speed.]
[v6 bright space version: copy of render_ckpt.py; physics/control loop UNCHANGED, only scene lighting/colour + foot projection]
Checkpoint rollout render: with an ONNX policy we trained OURSELVES (or a pretrained one),
5 s, 1080x1920, same camera path; first/last frame PNG + <out>.json (distance, trunk height, fell or not).

Open Duck Mini v2 walking with its pretrained ONNX policy (BEST_WALK_ONNX),
rendered headless in MuJoCo on the Mac in our video look: cosmic-black
sky, very dark graphite floor with a faint grid, ONE warm key light that rides
with the duck, a weak cool fill, slow camera dolly. Also writes per-frame
screen positions (output px) of the head and trunk (used by our video overlays).

Based on render_walk.py (same jax stubs, same control loop). Model files are
Open_Duck_Playground's open_duck_mini_v2 MJCF (the policy's observation needs
its sensors); only the scene around it is ours.

  venv/bin/python -I render_walk_v6.py <Open_Duck_Playground> <onnx> <out.mp4> \
      [seconds=8] [vx=0.15] [W=2160] [H=3840] [--stills]
"""
import sys, os, types, pathlib, json, tempfile
args = [a for a in sys.argv[1:] if not a.startswith("--")]
STILLS = "--stills" in sys.argv
repo, onnx_path, out = args[0], args[1], args[2]
secs = float(args[3]) if len(args) > 3 else 5.0
vx = float(args[4]) if len(args) > 4 else 0.15
W = int(args[5]) if len(args) > 5 else 1080
H = int(args[6]) if len(args) > 6 else 1920
sys.path.append(repo)
import numpy as np, mujoco, imageio

root = pathlib.Path(repo) / "playground/open_duck_mini_v2"
def get_assets():
    a = {}
    for p in (root / "xmls").glob("*.xml"): a[p.name] = p.read_bytes()
    for p in (root / "xmls/assets").glob("*"): a[p.name] = p.read_bytes()
    return a
base = types.ModuleType("playground.open_duck_mini_v2.base"); base.get_assets = get_assets
sys.modules["playground.open_duck_mini_v2.base"] = base
utils = types.ModuleType("playground.common.utils")
class LowPassActionFilter:
    def __init__(self, *a, **k): pass
utils.LowPassActionFilter = LowPassActionFilter
sys.modules["playground.common.utils"] = utils
import playground.open_duck_mini_v2 as pkg; pkg.base = base
from playground.open_duck_mini_v2.mujoco_infer import MjInfer

# Same physics as scene_flat_terrain.xml (floor geom, friction, home key);
# only visuals differ.
# Bright space look (luma target ~85+):
# dark navy sky gradient + navy-graphite floor + a light pool under the duck. Tunable via env.
E = lambda k, d: os.environ.get(k, d)
SKY1, SKY2 = E("SKY1", "0.27 0.33 0.60"), E("SKY2", "0.20 0.24 0.45")
FL1, FL2, FLM = E("FL1", "0.30 0.335 0.47"), E("FL2", "0.285 0.32 0.45"), E("FLM", "0.48 0.52 0.68")
FL_R = E("FL_R", "0.10")
ZFAR, ZNEAR = E("ZFAR", "40"), E("ZNEAR", "0.01")
FOG0, FOG1, FOGC = E("FOG0", "1.0"), E("FOG1", "3.6"), E("FOGC", "0.21 0.25 0.46")
SKYL = E("SKYL", "0.22 0.25 0.38")
POOL = E("POOL", "0.95 0.88 0.80")
HL_D, HL_A = float(E("HL_D", "0.14")), float(E("HL_A", "0.24"))
SCENE = f"""
<mujoco model="scene_v6_parlak">
  <include file="open_duck_mini_v2.xml"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="38"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight diffuse="{HL_D} {HL_D} {HL_D*1.1:.3f}" ambient="{HL_A} {HL_A*1.04:.3f} {HL_A*1.25:.3f}" specular="0 0 0"/>
    <rgba haze="0.20 0.24 0.45 1" fog="{FOGC} 1"/>
    <map znear="{ZNEAR}" zfar="{ZFAR}" haze="0.5" fogstart="{FOG0}" fogend="{FOG1}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="{SKY1}" rgb2="{SKY2}" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="{FL1}" rgb2="{FL2}"
             markrgb="{FLM}" width="400" height="400"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="6 6" reflectance="{FL_R}" specular="0.35" shininess="0.5"/>
  </asset>
  <worldbody>
    <light name="key" mode="trackcom" pos="-0.35 -0.75 0.85" dir="0.35 0.75 -0.85" diffuse="1.35 1.12 0.95" specular="0.45 0.35 0.28"
           castshadow="true" cutoff="40" exponent="6" attenuation="1 0 0"/>
    <light name="pool" mode="trackcom" pos="0 0 1.6" dir="0 0 -1" diffuse="{POOL}" specular="0 0 0"
           castshadow="false" cutoff="24" exponent="3" attenuation="1 0 0"/>
    <light name="sky" directional="true" pos="0 0 3" dir="0.15 0.35 -1" diffuse="{SKYL}" specular="0 0 0" castshadow="false"/>
    <light name="fill" mode="trackcom" pos="-0.8 0.2 0.5" dir="0.8 -0.2 -0.5" diffuse="0.34 0.39 0.56" specular="0 0 0" castshadow="false"/>
    <light name="rim" mode="trackcom" pos="0.6 0.45 0.7" dir="-0.6 -0.45 -0.7" diffuse="0.45 0.50 0.66" specular="0.2 0.2 0.25" castshadow="false"/>
    <body name="floor">
      <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
    </body>
  </worldbody>
  <contact>{(root / "xmls/hile_contact_pairs.txt").read_text()}
  </contact>
  <keyframe>
    <key name="home" qpos="0 0 0.15 1 0 0 0 0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.003 -0.065 0.635 1.379 -0.796"
         ctrl="0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.003 -0.065 0.635 1.379 -0.796"/>
  </keyframe>
</mujoco>
"""
tmp = pathlib.Path(tempfile.mkdtemp()) / "scene_v6.xml"
tmp.write_text(SCENE)

m = MjInfer(str(tmp), str(root / "data/polynomial_coefficients.pkl"), onnx_path, False)
m.commands[0] = vx
# STOCH=1: sample with exploration noise as in training (a = tanh(loc + std*eps)); the ONNX loc_out/scale outputs
STOCH = os.environ.get("STOCH", "0") == "1"
if STOCH:
    import onnxruntime as _ort
    class _Stoch:
        def __init__(self, path, seed):
            self.s = _ort.InferenceSession(path, providers=["CPUExecutionProvider"]); self.rng = np.random.default_rng(seed)
        def infer(self, obs):
            _, loc, sc = self.s.run(None, {"obs": np.asarray([obs], dtype=np.float32)})
            return np.tanh(loc[0] + sc[0] * self.rng.standard_normal(loc.shape[1])).astype(np.float32)
    m.policy = _Stoch(onnx_path, int(os.environ.get("SEED", 0)))
model, data = m.model, m.data
fps = 30
r = mujoco.Renderer(model, height=H, width=W)
if E("FOG", "1") == "1": r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
LOOK_DZ = float(os.environ.get("LOOK_DZ", -0.035))  # look below the trunk: duck sits high, floor below for type
head_id = model.body("head_assembly").id
trunk_id = model.body("trunk_assembly").id

import os
AZ0, AZ1 = float(os.environ.get("AZ0", 18)), float(os.environ.get("AZ1", 44))
ease = lambda u: u * u * (3 - 2 * u)
def cam_at(t):
    u = ease(min(1.0, max(0.0, t / secs)))
    cam.distance = 2.0 - 0.45 * u
    cam.azimuth = AZ0 + (AZ1 - AZ0) * u
    cam.elevation = -8 - 5 * u
    p = data.xpos[trunk_id]
    cam.lookat[:] = [p[0], p[1], p[2] + LOOK_DZ]

def project(p):
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


# zfar=40 (anything larger breaks the shadow map) clips the sky below the horizon -> black band at the horizon.
# Fill dark pixels whose depth is the "far plane" with the fog colour (background only; duck/floor unchanged).
FILL = np.array([float(x) * 255 for x in FOGC.split()], dtype=np.float32)
def render_frame():
    img = r.render()
    if E("FILLBAND", "0") != "1": return img
    r.enable_depth_rendering(); dep = r.render(); r.disable_depth_rendering()
    far = dep.max()
    bg = dep >= far * 0.999
    dark = img.astype(np.int32).sum(axis=2) < 150
    m = bg.copy()
    m[1:] |= bg[:-1]; m[:-1] |= bg[1:]; m[:, 1:] |= bg[:, :-1]; m[:, :-1] |= bg[:, 1:]
    m &= dark
    out = img.copy(); out[m] = FILL.astype(np.uint8)
    return out

steps_per_frame = int(round(1 / fps / m.sim_dt))
n_frames = int(secs * fps)
w = None if STILLS else imageio.get_writer(out, fps=fps, codec="libx264", quality=None, bitrate="60M", macro_block_size=8,
                                          ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "14"])
counter = 0
start_xy = data.qpos[:2].copy()
mujoco.mj_forward(model, data)
FOOT_Z0 = [float(data.site_xpos[model.site(n).id][2]) for n in ("left_foot", "right_foot")]
frames = []
FALL_Z = 0.09  # home trunk z=0.15; count as fallen when below this
min_z, fell_t = 1e9, None
# In the training env only the foot geoms collide with the floor (a fall ends the episode). So the falling duck
# doesn't pass through the floor in the render: once the trunk starts tipping (up-axis z < 0.6 or trunk z < 0.11)
# make the robot's visual meshes collide ONLY with the floor (contype=0, conaffinity=1 -> no self-contact).
# While walking upright the physics stays identical to training.
robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and model.geom_contype[g] == 0
               and model.geom_conaffinity[g] == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
collide_on_t = None
# Step counter: 1 step when a foot site lifts >1 cm off the floor and comes back down (to tell walking from stepping in place)
foot_sites = [model.site(n).id for n in ("left_foot", "right_foot")]
foot_up = [False, False]; foot_steps = [0, 0]; foot_zr = [[9, -9], [9, -9]]
print("render-only floor-collidable geoms:", len(robot_geoms))
flip_t, flipped_frames, min_up, tq_sq, both_air, counted, jmove, max_trunk_z = None, 0, 1.0, [], 0, 0, 0.0, 0.0
prev_q = data.qpos[7:].copy()
for f in range(n_frames):
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data); counter += 1
        if counter % m.decimation == 0:
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
    cam_at(f / fps)
    r.update_scene(data, cam)
    frames.append({"t": round(f / fps, 4), "head": project(data.xpos[head_id]), "trunk": project(data.xpos[trunk_id]), "lfoot": project(data.site_xpos[foot_sites[0]]), "rfoot": project(data.site_xpos[foot_sites[1]]), "fz": [round(float(data.site_xpos[s][2]), 4) for s in foot_sites],
                   "dist": round(float(np.linalg.norm(data.qpos[:2] - start_xy)), 4), "dx": round(float(data.qpos[0] - start_xy[0]), 4)})
    if STILLS:
        if f % 60 == 0 or f == n_frames - 1:
            imageio.imwrite(out.replace(".mp4", f"_f{f:03d}.png"), render_frame())
    else:
        img = render_frame(); w.append_data(img)
        if f == 0: imageio.imwrite(out.replace(".mp4", "_first.png"), img)
        if f == n_frames - 1: imageio.imwrite(out.replace(".mp4", "_last.png"), img)
    tz = float(data.qpos[2]); min_z = min(min_z, tz)
    upz = float(data.xmat[trunk_id].reshape(3, 3)[2, 2])
    if E("MESHCOL", "0") == "1" and collide_on_t is None and (upz < 0.6 or tz < 0.11):
        model.geom_conaffinity[robot_geoms] = 1; collide_on_t = round(f / fps, 3)
        if hasattr(model, "body_conaffinity"):  # the broadphase body filter must be updated too
            for g in robot_geoms: model.body_conaffinity[model.geom_bodyid[g]] |= 1
    for k, sid in enumerate(foot_sites):
        z = float(data.site_xpos[sid][2])
        if f < 10: continue  # first 1/3 s: the duck settles onto the floor from the home pose
        foot_zr[k] = [min(foot_zr[k][0], z), max(foot_zr[k][1], z)]
        base = foot_zr[k][0]  # lowest height so far (foot on the floor)
        if not foot_up[k] and z > base + 0.01: foot_up[k] = True
        elif foot_up[k] and z < base + 0.004: foot_up[k] = False; foot_steps[k] += 1
    if fell_t is None and (upz < 0.5 or tz < FALL_Z): fell_t = round(f / fps, 3)  # tipped >60°
    gz = float(data.sensor("upvector").data[2])  # the training termination sensor (<0: flipped more than 90°)
    if flip_t is None and gz < 0: flip_t = round(f / fps, 3)
    flipped_frames += gz < 0; min_up = min(min_up, gz)
    tq_sq.append(float(np.sum(np.square(data.actuator_force))))
    fz_now = [float(data.site_xpos[s_][2]) for s_ in foot_sites]
    if f >= 10: both_air += all(z > b[0] + 0.01 for z, b in zip(fz_now, foot_zr)); counted += 1
    jmove += float(np.sum(np.abs(data.qpos[7:] - prev_q))); prev_q = data.qpos[7:].copy()
    max_trunk_z = max(max_trunk_z, tz)
if w: w.close()
dist = float(np.linalg.norm(data.qpos[:2] - start_xy))
json.dump({"stochastic": STOCH, "seed": int(os.environ.get("SEED", 0)), "fps": fps, "width": W, "height": H, "vx": vx, "secs": secs, "onnx": onnx_path,
           "dist_m": round(dist, 4), "dx_m": round(float(data.qpos[0] - start_xy[0]), 4),
           "final_trunk_z": round(float(data.qpos[2]), 4), "min_trunk_z": round(min_z, 4), "fell_t": fell_t, "floor_collision_on_t": collide_on_t, "foot_steps": foot_steps, "foot_z_range": foot_zr, "foot_z0": FOOT_Z0,
           "flip_t": flip_t, "flipped_s": round(flipped_frames / fps, 3), "min_upz": round(min_up, 3), "max_trunk_z": round(max_trunk_z, 4),
           "torque_sq_mean": round(float(np.mean(tq_sq)), 4), "both_feet_air_frac": round(both_air / max(counted, 1), 4),
           "joint_travel_rad": round(jmove, 3), "foot_lift_max_m": [round(b[1] - b[0], 4) for b in foot_zr],
           "frames": frames}, open(out.replace(".mp4", ".json"), "w"))
print("frames", n_frames, "dist_m", float(np.linalg.norm(data.qpos[:2] - start_xy)), "trunk_z", float(data.qpos[2]))
