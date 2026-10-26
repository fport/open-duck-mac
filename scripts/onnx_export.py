"""Convert Brax PPO policy parameters (normalizer + MLP) to ONNX; no TensorFlow / tf2onnx needed.

Used as a library by train_mac.py (an ONNX file is written at every checkpoint). As a script it
re-exports a saved checkpoint (ckpt_*.pkl, written by train_mac.py) to ONNX:

  python -I scripts/onnx_export.py runs/run1/ckpt/ckpt_072437760.pkl out.onnx [--act_size 14]

Output has the same interface as Open_Duck_Playground's export_onnx.py: input "obs" (1, obs_size),
output "continuous_actions" = tanh(loc), loc = first half of the MLP output; swish in hidden layers.
Normalisation: (obs - mean) / std (brax running_statistics, std already clipped).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def export(normalizer_mean, normalizer_std, layers, act_size, out_path):
    """layers: [(kernel[in,out], bias[out]), ...] — brax MLP hidden_0..N in order."""
    obs_size = int(np.asarray(normalizer_mean).shape[-1])
    inits, nodes = [], []
    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))
    const("mean", normalizer_mean); const("std", normalizer_std)
    nodes.append(helper.make_node("Sub", ["obs", "mean"], ["x_c"]))
    nodes.append(helper.make_node("Div", ["x_c", "std"], ["x0"]))
    x = "x0"
    for i, (k, b) in enumerate(layers):
        const(f"W{i}", k); const(f"b{i}", b)
        nodes.append(helper.make_node("Gemm", [x, f"W{i}", f"b{i}"], [f"h{i}"]))
        x = f"h{i}"
        if i < len(layers) - 1:
            nodes.append(helper.make_node("Sigmoid", [x], [f"s{i}"]))
            nodes.append(helper.make_node("Mul", [x, f"s{i}"], [f"a{i}"]))
            x = f"a{i}"
    for n, v in (("starts", [0]), ("ends", [act_size]), ("axes", [1]),
                 ("starts2", [act_size]), ("ends2", [2 * act_size])):
        inits.append(numpy_helper.from_array(np.array(v, dtype=np.int64), n))
    nodes.append(helper.make_node("Slice", [x, "starts", "ends", "axes"], ["loc"]))
    nodes.append(helper.make_node("Tanh", ["loc"], ["continuous_actions"]))
    # 2nd output (for exploration noise): brax NormalTanhDistribution std = softplus(raw) + 0.001
    const("min_std", [0.001])
    nodes.append(helper.make_node("Slice", [x, "starts2", "ends2", "axes"], ["raw_scale"]))
    nodes.append(helper.make_node("Softplus", ["raw_scale"], ["sp"]))
    nodes.append(helper.make_node("Add", ["sp", "min_std"], ["scale"]))
    nodes.append(helper.make_node("Identity", ["loc"], ["loc_out"]))
    g = helper.make_graph(nodes, "duck_policy",
                          [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_size])],
                          [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, act_size]),
                           helper.make_tensor_value_info("loc_out", TensorProto.FLOAT, [1, act_size]),
                           helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1, act_size])],
                          inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)


def from_brax_params(params, act_size, out_path):
    """params = (normalizer_params, policy_params, value_params) — brax ppo policy_params_fn argument."""
    norm, pol = params[0], params[1]
    mean = np.asarray(norm.mean["state"] if isinstance(norm.mean, dict) else norm.mean)
    std = np.asarray(norm.std["state"] if isinstance(norm.std, dict) else norm.std)
    p = pol["params"] if "params" in pol else pol
    mlp = p["MLP_0"] if "MLP_0" in p else p
    names = sorted([k for k in mlp if k.startswith("hidden_")], key=lambda s: int(s.split("_")[1]))
    layers = [(np.asarray(mlp[n]["kernel"]), np.asarray(mlp[n]["bias"])) for n in names]
    export(mean, std, layers, act_size, out_path)
    return layers


def main():
    import argparse, pickle
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="ckpt_*.pkl written by train_mac.py (numpy pytree: normalizer, policy, value)")
    ap.add_argument("out", help="output .onnx path")
    ap.add_argument("--act_size", type=int, default=14, help="action size (Open Duck Mini v2: 14)")
    a = ap.parse_args()
    import brax.training.acme.running_statistics  # noqa: F401  (class of the pickled normalizer state)
    with open(a.pkl, "rb") as f:
        params = pickle.load(f)
    layers = from_brax_params(params, a.act_size, a.out)
    print(f"wrote {a.out}: obs {int(np.asarray(params[0].mean['state'] if isinstance(params[0].mean, dict) else params[0].mean).shape[-1])}"
          f" -> MLP {[k.shape[1] for k, _ in layers]} -> {a.act_size} actions")


if __name__ == "__main__":
    main()
