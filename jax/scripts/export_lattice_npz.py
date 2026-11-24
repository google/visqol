"""Export a ViSQOL TF-Lattice `.tflite` model to a JAX-compatible `.npz` file.

Reads a TensorFlow Lattice `CalibratedLatticeEnsemble` `.tflite` file via the
TensorFlow Lite interpreter, traces the TFLite operator graph from the `predict`
signature without relying on hardcoded tensor names, verifies numerical parity
against live TFLite inference, and writes a compressed `.npz` archive loadable
by `visqol_jax.lattice.load_lattice_model`.
"""

import pathlib
from typing import Any

import click
import numpy as np
import tensorflow as tf

_FEATURE_PREFIXES = ("fvnsim", "fvnsim10_", "fstdnsim", "fvdegenergy")


def _load_undelegated_interpreter(
    tflite_path: pathlib.Path,
) -> tf.lite.Interpreter:
  """Load a TFLite interpreter without delegates so raw ops stay inspectable."""
  interp = tf.lite.Interpreter(
      model_path=str(tflite_path),
      experimental_op_resolver_type=(
          tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
      ),
  )
  interp.allocate_tensors()
  return interp


class _TFLiteGraph:
  """Helper for inspecting tensors, constants, and op dataflow in a TFLite."""

  def __init__(self, interp: tf.lite.Interpreter) -> None:
    self.interp = interp
    self.tensors: dict[int, dict[str, Any]] = {
        int(t["index"]): t for t in interp.get_tensor_details()
    }
    self.producers: dict[int, dict[str, Any]] = {}
    for op in interp._get_ops_details():  # pylint: disable=protected-access
      for out_idx in op["outputs"]:
        if int(out_idx) >= 0:
          self.producers[int(out_idx)] = op

  def is_const(self, idx: int) -> bool:
    return idx not in self.producers

  def const(self, idx: int) -> np.ndarray:
    if not self.is_const(idx):
      raise ValueError(f"Tensor {idx} ({self.name(idx)}) is not a constant")
    return np.array(self.interp.get_tensor(idx), copy=True)

  def name(self, idx: int) -> str:
    return str(self.tensors[idx]["name"])

  def shape(self, idx: int) -> tuple[int, ...]:
    return tuple(int(d) for d in self.tensors[idx]["shape"])

  def producer(self, idx: int) -> dict[str, Any]:
    if idx not in self.producers:
      raise ValueError(f"Tensor {idx} ({self.name(idx)}) has no producer op")
    return self.producers[idx]

  def valid_inputs(self, op: dict[str, Any]) -> list[int]:
    return [int(i) for i in op["inputs"] if int(i) >= 0]

  def strip_view_ops(self, idx: int) -> int:
    while idx in self.producers and self.producers[idx]["op_name"] in (
        "RESHAPE",
        "EXPAND_DIMS",
        "SQUEEZE",
    ):
      idx = int(self.producers[idx]["inputs"][0])
    return idx


def _feature_names(num_bands: int) -> list[str]:
  names: list[str] = []
  for prefix in _FEATURE_PREFIXES:
    for band in range(num_bands):
      names.append(f"{prefix}{band}")
  names.append("tau")
  return names


def _parse_pwl_calibrator(
    g: _TFLiteGraph, matmul_idx: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
  """Parse a PWL calibrator `FULLY_CONNECTED` output into arrays."""
  fc_op = g.producer(matmul_idx)
  if fc_op["op_name"] != "FULLY_CONNECTED":
    raise ValueError(f"Expected FULLY_CONNECTED, got {fc_op['op_name']}")
  fc_ins = g.valid_inputs(fc_op)
  concat_idx, w_idx = fc_ins[0], fc_ins[1]
  weights = np.asarray(g.const(w_idx), dtype=np.float32)
  if weights.ndim != 2:
    raise ValueError(f"Expected 2D calibrator weights, got {weights.shape}")

  # CONCATENATION([ones, clipped_slopes]) -> MAXIMUM(MINIMUM(slopes, 1), 0)
  concat_op = g.producer(concat_idx)
  if concat_op["op_name"] != "CONCATENATION":
    raise ValueError(f"Expected CONCATENATION, got {concat_op['op_name']}")
  c_ins = g.valid_inputs(concat_op)
  if len(c_ins) != 2:
    raise ValueError(
        f"Expected 2 concat inputs in calibrator, got {len(c_ins)}"
    )
  slopes_idx = c_ins[1]

  # Unwrap clamp-to-[0, 1] ops (RELU/MAXIMUM and MINIMUM, or RELU_0_TO_1).
  while slopes_idx in g.producers and g.producer(slopes_idx)["op_name"] in (
      "MAXIMUM",
      "MINIMUM",
      "RELU",
      "RELU_0_TO_1",
  ):
    op = g.producer(slopes_idx)
    non_const = [i for i in g.valid_inputs(op) if not g.is_const(i)]
    if len(non_const) != 1:
      raise ValueError(f"Expected 1 non-const input to {op['op_name']}")
    slopes_idx = non_const[0]

  scale_op = g.producer(slopes_idx)
  scale_ins = g.valid_inputs(scale_op)
  sub_idx = next(i for i in scale_ins if not g.is_const(i))
  scale_const = np.asarray(
      g.const(next(i for i in scale_ins if g.is_const(i))), dtype=np.float32
  ).reshape(-1)
  if scale_op["op_name"] == "DIV":
    inv_lengths = (np.float32(1.0) / scale_const).astype(np.float32)
  elif scale_op["op_name"] == "MUL":
    inv_lengths = scale_const.astype(np.float32)
  else:
    raise ValueError(
        f"Expected DIV or MUL in calibrator, got {scale_op['op_name']}"
    )

  sub_op = g.producer(sub_idx)
  if sub_op["op_name"] != "SUB":
    raise ValueError(f"Expected SUB in calibrator, got {sub_op['op_name']}")
  sub_ins = g.valid_inputs(sub_op)
  x_idx = next(i for i in sub_ins if not g.is_const(i))
  keypoints = np.asarray(
      g.const(next(i for i in sub_ins if g.is_const(i))), dtype=np.float32
  ).reshape(-1)
  return x_idx, keypoints, inv_lengths, weights


def _collect_lattice_dim_inputs(g: _TFLiteGraph, idx: int) -> list[int]:
  """Return ordered calibrated scalar tensor indices feeding a lattice."""
  idx = g.strip_view_ops(idx)
  op = g.producer(idx)
  if op["op_name"] in ("MUL", "BATCH_MATMUL"):
    a, b = g.valid_inputs(op)
    sa, sb = g.shape(a), g.shape(b)
    # Kronecker step multiplies (1, 2^k, 1) left by (1, 1, 2) right.
    if len(sa) == 3 and len(sb) == 3 and sa[2] == 1 and sb[1] == 1:
      left, right = a, b
    elif len(sa) == 3 and len(sb) == 3 and sb[2] == 1 and sa[1] == 1:
      left, right = b, a
    else:
      raise ValueError(f"Unexpected lattice product shapes {sa} and {sb}")
    return _collect_lattice_dim_inputs(g, left) + _collect_lattice_dim_inputs(
        g, right
    )

  # Leaf: 1 - min(abs(c - [0, 1]), 1)
  if op["op_name"] != "SUB":
    raise ValueError(
        f"Expected SUB at lattice weight leaf, got {op['op_name']}"
    )
  min_idx = next(i for i in g.valid_inputs(op) if not g.is_const(i))
  min_op = g.producer(min_idx)
  if min_op["op_name"] != "MINIMUM":
    raise ValueError(
        f"Expected MINIMUM in lattice leaf, got {min_op['op_name']}"
    )
  abs_idx = next(i for i in g.valid_inputs(min_op) if not g.is_const(i))
  abs_op = g.producer(abs_idx)
  if abs_op["op_name"] != "ABS":
    raise ValueError(f"Expected ABS in lattice leaf, got {abs_op['op_name']}")
  sub0_idx = g.valid_inputs(abs_op)[0]
  sub0_op = g.producer(sub0_idx)
  if sub0_op["op_name"] != "SUB":
    raise ValueError(f"Expected SUB in lattice leaf, got {sub0_op['op_name']}")
  c_idx = next(i for i in g.valid_inputs(sub0_op) if not g.is_const(i))
  return [c_idx]


def _trace_calibrated_scalar(
    g: _TFLiteGraph, c_idx: int
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, bool, float, float]:
  """Trace a lattice dimension input back to its input feature calibrator."""
  split_idx = 0
  cur = c_idx
  op = g.producer(cur)
  if op["op_name"] in ("SPLIT", "SPLIT_V"):
    split_idx = [int(o) for o in op["outputs"]].index(cur)
    cur = int(op["inputs"][1] if op["op_name"] == "SPLIT" else op["inputs"][0])
    op = g.producer(cur)

  has_missing = False
  missing_val = 0.0
  missing_output = 0.0
  if op["op_name"] == "ADD":
    # Missing-value blend: MUL(cast_eq, miss_out_var) + MUL(1 - cast_eq, fc_out)
    in0, in1 = g.valid_inputs(op)
    fc_out_idx: int | None = None
    for branch in (in0, in1):
      b_op = g.producer(branch)
      if b_op["op_name"] != "MUL":
        raise ValueError(
            f"Expected MUL in missing-value ADD, got {b_op['op_name']}"
        )
      b_ins = g.valid_inputs(b_op)
      consts = [i for i in b_ins if g.is_const(i)]
      if consts:
        has_missing = True
        miss_vec = np.asarray(g.const(consts[0]), dtype=np.float32).reshape(-1)
        missing_output = float(miss_vec[split_idx])
        cast_idx = next(i for i in b_ins if not g.is_const(i))
        cast_op = g.producer(cast_idx)
        if cast_op["op_name"] != "CAST":
          raise ValueError(
              f"Expected CAST in missing branch, got {cast_op['op_name']}"
          )
        eq_op = g.producer(g.valid_inputs(cast_op)[0])
        if eq_op["op_name"] != "EQUAL":
          raise ValueError(
              f"Expected EQUAL in missing branch, got {eq_op['op_name']}"
          )
        mval_idx = next(i for i in g.valid_inputs(eq_op) if g.is_const(i))
        missing_val = float(
            np.asarray(g.const(mval_idx), dtype=np.float32).reshape(-1)[0]
        )
      else:
        for sub_in in b_ins:
          if g.producer(sub_in)["op_name"] == "FULLY_CONNECTED":
            fc_out_idx = sub_in
    if not has_missing or fc_out_idx is None:
      raise ValueError("Could not parse missing-value ADD structure")
    cur = fc_out_idx

  x_idx, keypoints, inv_lengths, cal_w_2d = _parse_pwl_calibrator(g, cur)
  root_idx = g.strip_view_ops(x_idx)
  return (
      root_idx,
      split_idx,
      keypoints,
      inv_lengths,
      cal_w_2d[split_idx],
      has_missing,
      missing_val,
      missing_output,
  )


def extract_lattice_arrays(tflite_path: pathlib.Path) -> dict[str, np.ndarray]:
  """Extract `LatticeModel` NumPy arrays from a TF-Lattice `.tflite` file."""
  interp = _load_undelegated_interpreter(tflite_path)
  g = _TFLiteGraph(interp)

  runner = interp.get_signature_runner("predict")
  in_details = runner.get_input_details()
  out_details = runner.get_output_details()
  if "predictions" not in out_details:
    raise ValueError(
        "Expected 'predictions' output in 'predict' signature, got"
        f" {list(out_details)}"
    )

  nf = len(in_details)
  if nf < 5 or (nf - 1) % 4 != 0:
    raise ValueError(f"Expected 4 * num_bands + 1 predict inputs, got {nf}")
  num_bands = (nf - 1) // 4
  feature_names = _feature_names(num_bands)
  missing_inputs = [f for f in feature_names if f not in in_details]
  if missing_inputs:
    raise ValueError(f"Missing expected predict inputs: {missing_inputs[:5]}")
  input_idx_to_feat = {
      int(in_details[name]["index"]): f_idx
      for f_idx, name in enumerate(feature_names)
  }

  pred_idx = int(out_details["predictions"]["index"])
  z_idx, out_kp, out_inv_len, out_w_2d = _parse_pwl_calibrator(g, pred_idx)
  if out_w_2d.shape[0] != 1:
    raise ValueError(f"Expected scalar output calibrator, got {out_w_2d.shape}")
  out_w = out_w_2d[0]

  ens_op = g.producer(z_idx)
  if ens_op["op_name"] != "FULLY_CONNECTED":
    raise ValueError(
        f"Expected ensemble FULLY_CONNECTED, got {ens_op['op_name']}"
    )
  ens_ins = g.valid_inputs(ens_op)
  ens_concat_idx, ens_w_idx = ens_ins[0], ens_ins[1]
  ens_w_2d = np.asarray(g.const(ens_w_idx), dtype=np.float32)
  if ens_w_2d.shape[0] != 1:
    raise ValueError(f"Expected (1, nl) ensemble weights, got {ens_w_2d.shape}")
  ensemble_weights = ens_w_2d[0]

  ens_concat_op = g.producer(ens_concat_idx)
  if ens_concat_op["op_name"] == "CONCATENATION":
    lat_out_indices = g.valid_inputs(ens_concat_op)
  else:
    lat_out_indices = [ens_concat_idx]
  nl = len(lat_out_indices)
  if ensemble_weights.shape != (nl,):
    raise ValueError(
        f"Ensemble weight length {ensemble_weights.shape} != {nl} lattices"
    )

  in_kp_map: dict[int, np.ndarray] = {}
  in_inv_map: dict[int, np.ndarray] = {}
  in_has_miss = np.zeros((nf,), dtype=np.bool_)
  in_miss_val = np.zeros((nf,), dtype=np.float32)

  feat_indices_rows: list[list[int]] = []
  cal_w_rows: list[list[np.ndarray]] = []
  cal_miss_rows: list[list[float]] = []
  lat_w_rows: list[np.ndarray] = []

  for lat_out_idx in lat_out_indices:
    lat_fc = g.producer(lat_out_idx)
    if lat_fc["op_name"] != "FULLY_CONNECTED":
      raise ValueError(
          f"Expected lattice FULLY_CONNECTED, got {lat_fc['op_name']}"
      )
    lat_ins = g.valid_inputs(lat_fc)
    kron_idx, lat_w_idx = lat_ins[0], lat_ins[1]
    lat_w = np.asarray(g.const(lat_w_idx), dtype=np.float32)
    if lat_w.shape[0] != 1:
      raise ValueError(
          f"Expected (1, 2**rank) lattice weights, got {lat_w.shape}"
      )
    lat_w_rows.append(lat_w[0])

    dim_c_indices = _collect_lattice_dim_inputs(g, kron_idx)
    if lat_w.shape[1] != (1 << len(dim_c_indices)):
      raise ValueError(
          f"Lattice weight width {lat_w.shape[1]} != 2**{len(dim_c_indices)}"
      )

    row_fidx: list[int] = []
    row_cal_w: list[np.ndarray] = []
    row_cal_miss: list[float] = []
    for c_idx in dim_c_indices:
      root_idx, _, kp, inv_l, cal_w, has_m, m_val, m_out = (
          _trace_calibrated_scalar(g, c_idx)
      )
      if root_idx not in input_idx_to_feat:
        raise ValueError(
            f"Calibrator root tensor {root_idx} ({g.name(root_idx)}) is not a"
            " predict signature input"
        )
      f_idx = input_idx_to_feat[root_idx]
      if f_idx in in_kp_map:
        if not np.array_equal(in_kp_map[f_idx], kp) or not np.array_equal(
            in_inv_map[f_idx], inv_l
        ):
          raise ValueError(
              f"Inconsistent keypoints for feature {feature_names[f_idx]}"
          )
      else:
        in_kp_map[f_idx] = kp
        in_inv_map[f_idx] = inv_l
      if has_m:
        in_has_miss[f_idx] = True
        in_miss_val[f_idx] = np.float32(m_val)
      row_fidx.append(f_idx)
      row_cal_w.append(cal_w)
      row_cal_miss.append(m_out)

    feat_indices_rows.append(row_fidx)
    cal_w_rows.append(row_cal_w)
    cal_miss_rows.append(row_cal_miss)

  in_k = next(iter(in_kp_map.values())).shape[0]
  in_keypoints = np.zeros((nf, in_k), dtype=np.float32)
  in_inv_lengths = np.ones((nf, in_k), dtype=np.float32)
  for f_idx in range(nf):
    if f_idx in in_kp_map:
      in_keypoints[f_idx] = in_kp_map[f_idx]
      in_inv_lengths[f_idx] = in_inv_map[f_idx]

  return {
      "in_keypoints": in_keypoints,
      "in_inv_lengths": in_inv_lengths,
      "in_has_missing": in_has_miss,
      "in_missing_value": in_miss_val,
      "lattice_feature_indices": np.asarray(feat_indices_rows, dtype=np.int32),
      "lattice_calibrator_weights": np.asarray(cal_w_rows, dtype=np.float32),
      "lattice_calibrator_missing_output": np.asarray(
          cal_miss_rows, dtype=np.float32
      ),
      "lattice_weights": np.asarray(lat_w_rows, dtype=np.float32),
      "ensemble_weights": ensemble_weights.astype(np.float32),
      "out_keypoints": out_kp.astype(np.float32),
      "out_inv_lengths": out_inv_len.astype(np.float32),
      "out_weights": out_w.astype(np.float32),
  }


def _eval_lattice_np(arrays: dict[str, np.ndarray], x: np.ndarray) -> float:
  """Pure-NumPy `float32` evaluation matching `lattice.eval_lattice`."""
  x32 = np.asarray(x, dtype=np.float32)
  slopes = np.clip(
      (x32[:, None] - arrays["in_keypoints"]) * arrays["in_inv_lengths"],
      np.float32(0.0),
      np.float32(1.0),
  )
  basis = np.concatenate(
      [np.ones((x32.shape[0], 1), dtype=np.float32), slopes], axis=1
  )
  fidx = arrays["lattice_feature_indices"]
  c = np.sum(basis[fidx] * arrays["lattice_calibrator_weights"], axis=-1)
  is_miss = arrays["in_has_missing"] & (x32 == arrays["in_missing_value"])
  c = np.where(is_miss[fidx], arrays["lattice_calibrator_missing_output"], c)

  one = np.float32(1.0)
  w0 = one - np.minimum(np.abs(c), one)
  w1 = one - np.minimum(np.abs(c - one), one)
  cur = arrays["lattice_weights"]
  for d in range(c.shape[1]):
    cur = cur.reshape(cur.shape[0], 2, -1)
    cur = cur[:, 0, :] * w0[:, d, None] + cur[:, 1, :] * w1[:, d, None]
  z = np.sum(cur[:, 0] * arrays["ensemble_weights"], dtype=np.float32)

  out_slopes = np.clip(
      (z - arrays["out_keypoints"]) * arrays["out_inv_lengths"],
      np.float32(0.0),
      np.float32(1.0),
  )
  out_basis = np.concatenate([np.ones((1,), dtype=np.float32), out_slopes])
  return float(np.sum(out_basis * arrays["out_weights"], dtype=np.float32))


def verify_against_tflite(
    tflite_path: pathlib.Path,
    arrays: dict[str, np.ndarray],
    *,
    num_samples: int = 8,
    atol: float = 1e-5,
) -> float:
  """Verify extracted `.npz` arrays against live TFLite inference."""
  interp = _load_undelegated_interpreter(tflite_path)
  runner = interp.get_signature_runner("predict")
  nf = arrays["in_keypoints"].shape[0]
  num_bands = (nf - 1) // 4
  feature_names = _feature_names(num_bands)

  rng = np.random.default_rng(0)
  probes: list[np.ndarray] = [
      np.ones((nf,), dtype=np.float32),
      np.zeros((nf,), dtype=np.float32),
      np.where(
          arrays["in_has_missing"],
          arrays["in_missing_value"],
          np.float32(0.5),
      ).astype(np.float32),
  ]
  for _ in range(num_samples):
    x = rng.uniform(0.0, 1.0, size=(nf,)).astype(np.float32)
    x[3 * num_bands : 4 * num_bands] *= np.float32(20.0)
    x[-1] = np.float32(0.5)
    probes.append(x)

  max_err = 0.0
  for x in probes:
    feeds = {
        name: np.array([x[i]], dtype=np.float32)
        for i, name in enumerate(feature_names)
    }
    want = float(runner(**feeds)["predictions"].reshape(-1)[0])
    got = _eval_lattice_np(arrays, x)
    max_err = max(max_err, abs(got - want))
    if abs(got - want) > atol:
      raise ValueError(
          f"Parity check failed: extracted={got}, tflite={want} (atol={atol})"
      )
  return max_err


@click.command()
@click.argument(
    "tflite_path",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "output_path", type=click.Path(dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--verify/--no-verify",
    default=True,
    show_default=True,
    help="Verify extracted weights against live TFLite inference.",
)
def main(
    tflite_path: pathlib.Path, output_path: pathlib.Path, verify: bool
) -> None:
  """Convert a ViSQOL TF-Lattice `.tflite` model to a compressed `.npz`."""
  arrays = extract_lattice_arrays(tflite_path)
  if verify:
    max_err = verify_against_tflite(tflite_path, arrays)
    click.echo(f"Verified against TFLite (max abs diff: {max_err:.2e})")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **arrays)
  click.echo(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
  main()
