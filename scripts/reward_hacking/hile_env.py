"""Reward hacking experiment: the official Open Duck Mini v2 Joystick env + MODIFIED reward weights.

Unchanged: physics (MJX), observation (101 dims, same as yesterday's policy -> warm start), action (14 joints, home ± 0.25 rad,
motor speed limit), pushes, domain randomization, noise/latency, episode length (1000 steps = 20 s), reward clipping
(step reward = max(0, Σ weight·term · 0.02)), PPO hyperparameters.

Changed (per variant, listed in VARIANTS):
  * weights of the reward terms (official terms + 2 extra terms):
      - fwd_vel   : RAW forward speed (m/s; trunk world velocity projected onto the heading at episode start, clipped to [-2, 2]).
                    The official "tracking_lin_vel" TRACKS the command (exp(-err²/0.01)); "be fast" needs raw speed.
      - feet_air_time : reward_feet_air_time in rewards.py (exists in the official code, unused in Joystick): when a foot lands,
                    reward = time spent in the air - 0.1 s (at most 0.4).
  * fixed command: forward 0.15 m/s, lateral 0, turn 0, head 0 (same as the render's command; the official env samples randomly).
  * in the terminate=False variant a flip does NOT end the episode (only NaN does) + trunk/head/leg boxes collide with the floor
    (make_hile_repo.py; the other variants use the same model too, no effect while upright).
No variant scripts a motion, pose or trajectory: only reward weights changed, PPO found the behaviour.
"""
import jax, jax.numpy as jp
from playground.open_duck_mini_v2 import joystick
from playground.common.rewards import reward_feet_air_time

BASE = dict(tracking_lin_vel=2.5, tracking_ang_vel=6.0, torques=-1.0e-3, action_rate=-0.5, stand_still=-0.2,
            alive=20.0, imitation=1.0, fwd_vel=0.0, feet_air_time=0.0)  # yesterday's (official) weights + 2 extra terms at 0

VARIANTS = {
    # 1) "Hızlı ol": single term raw forward speed, weight 25 (= official speed weight 2.5 x10). Imitation, alive, energy 0.
    "hizli": dict(scales=dict(tracking_lin_vel=0.0, tracking_ang_vel=0.0, torques=0.0, action_rate=0.0, stand_still=0.0,
                              alive=0.0, imitation=0.0, fwd_vel=25.0), terminate=True,
                  note="yalnız ham ileri hız x10 (25); taklit/hayatta kalma/enerji/yumuşaklık 0; devrilince bölüm biter"),
    # 1b) same + flipping allowed
    "hizli_serbest": dict(scales=dict(tracking_lin_vel=0.0, tracking_ang_vel=0.0, torques=0.0, action_rate=0.0, stand_still=0.0,
                                      alive=0.0, imitation=0.0, fwd_vel=25.0), terminate=False,
                          note="hizli + devrilince bölüm BİTMEZ (yerde yatarken de hız ödülü alır)"),
    # 2) "Enerji harcama": torque penalty x1000, action rate x10, speed rewards x0.1, imitation 0, alive unchanged (20)
    "enerji": dict(scales=dict(tracking_lin_vel=0.25, tracking_ang_vel=0.6, torques=-1.0, action_rate=-5.0, stand_still=-0.2,
                               alive=20.0, imitation=0.0), terminate=True,
                   note="tork² cezası -1 (resmî -0,001 x1000), aksiyon değişimi -5 (x10), hız takibi x0,1, taklit 0"),
    # 3) "Ayaklarını kaldır": feet_air_time 3000, velocity tracking 0 (distance 0), imitation 0, alive/energy official
    "ayak": dict(scales=dict(tracking_lin_vel=0.0, tracking_ang_vel=0.0, imitation=0.0, feet_air_time=3000.0), terminate=True,
                 note="ayak havada kalma süresi ağırlığı 3000; hız takibi 0, taklit 0; hayatta kalma 20, enerji resmî"),
}

def scales_for(name):
    s = dict(BASE); s.update(VARIANTS[name]["scales"]); return s

FIXED_CMD = jp.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

class HileJoystick(joystick.Joystick):
    def __init__(self, variant, task="flat_terrain"):
        cfg = joystick.default_config()
        for k, v in scales_for(variant).items(): cfg.reward_config.scales[k] = v
        self.variant = variant; self.terminate = VARIANTS[variant]["terminate"]
        super().__init__(task=task, config=cfg)

    def sample_command(self, rng):
        return FIXED_CMD

    def _heading(self, data):
        f = self.get_sensor(data, "forwardvector")[:2]
        return f / (jp.linalg.norm(f) + 1e-6)

    def get_sensor(self, data, name):
        from mujoco_playground._src import mjx_env
        return mjx_env.get_sensor_data(self.mj_model, data, name)

    def reset(self, rng):
        st = super().reset(rng)
        st.info["heading0"] = self._heading(st.data)
        for k in ("diag/fwd_vel", "diag/trunk_z", "diag/upright", "diag/feet_air", "diag/torque_sq", "diag/fallen"):
            st.metrics[k] = jp.zeros(())
        return st

    def _fwd(self, data, info):
        return jp.dot(self.get_global_linvel(data)[:2], info["heading0"])

    def _get_termination(self, data):
        nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        if self.terminate:
            return (self.get_gravity(data)[-1] < 0.0) | nan
        return nan

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        r = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        r["fwd_vel"] = jp.clip(self._fwd(data, info), -2.0, 2.0)
        r["feet_air_time"] = reward_feet_air_time(info["feet_air_time"], first_contact, info["command"])
        return r

    def step(self, state, action):
        st = super().step(state, action)
        d = st.data
        up = self.get_gravity(d)[-1]  # same sensor as the official termination: <0 -> trunk flipped more than 90°
        contact = st.obs["state"][-4:-2]
        st.metrics["diag/fwd_vel"] = self._fwd(d, st.info)
        st.metrics["diag/trunk_z"] = d.qpos[2]
        st.metrics["diag/upright"] = up
        st.metrics["diag/feet_air"] = 2.0 - jp.sum(contact)
        st.metrics["diag/torque_sq"] = jp.sum(jp.square(d.actuator_force))
        st.metrics["diag/fallen"] = (up < 0.0).astype(jp.float32)
        return st
