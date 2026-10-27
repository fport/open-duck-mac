"""Makes a COPY of Open_Duck_Playground and only extends collision (for the reward hacking experiment).

In the official model only the two foot soles (mesh) collide with the floor: a falling duck's trunk passes through the floor
(not a problem in training since a flip ends the episode). So the duck can lie on the floor in the "flip allowed" variant,
invisible boxes (group 4) are added to the trunk, head and 4 leg parts; the boxes carry no mass (every body has an <inertial>,
unchanged mass is checked) and collide ONLY with the floor via <pair> (not with each other / the feet).
Box sizes are computed from each body's visual-mesh bounding box in the home pose.

  python -I make_hile_repo.py <Open_Duck_Playground> <Playground_hile>
"""
import sys, shutil, pathlib, re
import numpy as np, mujoco
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
X = dst / "playground/open_duck_mini_v2/xmls"
BODIES = ["trunk_assembly", "head_assembly", "knee_and_ankle_assembly", "knee_and_ankle_assembly_2",
          "knee_and_ankle_assembly_3", "knee_and_ankle_assembly_4"]
m = mujoco.MjModel.from_xml_path(str(src / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"))
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
boxes = {}
for bname in BODIES:
    b = m.body(bname).id; L = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        L.append((w - d.xpos[b]) @ d.xmat[b].reshape(3, 3))  # body local frame
    V = np.vstack(L); lo, hi = V.min(0), V.max(0)
    boxes[bname] = ((lo + hi) / 2, np.maximum((hi - lo) / 2 * 0.95, 0.005))
xml = (X / "open_duck_mini_v2.xml").read_text()
for bname, (c, h) in boxes.items():
    tag = re.search(rf'<body name="{bname}"[^>]*>', xml)
    geom = (f'\n<geom name="hc_{bname}" type="box" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
            f'size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" contype="0" conaffinity="0" group="4" rgba="1 0 0 0.3"/>')
    xml = xml[:tag.end()] + geom + xml[tag.end():]
(X / "open_duck_mini_v2.xml").write_text(xml)
pairs = "".join(f'\n    <pair geom1="floor" geom2="hc_{b}" condim="3" friction="0.6 0.6 0.005 0.0001 0.0001"/>' for b in BODIES)
for scene in ("scene_flat_terrain.xml",):
    s = (X / scene).read_text()
    i = s.rindex("</worldbody>") + len("</worldbody>")  # the first </worldbody> is inside a comment line
    s = s[:i] + "\n  <contact>" + pairs + "\n  </contact>" + s[i:]
    (X / scene).write_text(s)
(X / "hile_contact_pairs.txt").write_text(pairs)
m2 = mujoco.MjModel.from_xml_path(str(X / "scene_flat_terrain.xml"))
assert np.allclose(m.body_mass, m2.body_mass), "kütle değişti!"
assert np.allclose(m.body_inertia, m2.body_inertia), "eylemsizlik değişti!"
print("ok", {k: (np.round(c, 4).tolist(), np.round(h, 4).tolist()) for k, (c, h) in boxes.items()}, "npair", m2.npair)
