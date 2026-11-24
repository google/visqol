"""Tests for lattice.py speech lattice mapper against C++ ViSQOL v3.3.3."""

from collections.abc import Mapping
import dataclasses
import io
import os
import pathlib
import struct
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from visqol_jax import lattice
from visqol_jax import mappers

# CA01_01 clean speech identity degraded energy from conformance speech mode.
_CA01_PERFECT_DEGENERGY = (
    1.0629700787321201,
    3.120915321746017,
    10.25139156353318,
    16.18457718700952,
    12.588144485813276,
    16.680956935117315,
    16.922956648313267,
    14.656267686932688,
    11.543302998948448,
    11.572032864828918,
    11.697103695497272,
    12.044119880962622,
    12.728961668260084,
    10.669581952789695,
    9.048814557668258,
    9.07096331956905,
    8.949895703178681,
    9.701779102713129,
    8.691375166718226,
    6.517450019414028,
    7.126367059859797,
)

# CA01_01 vs transcoded_CA01_01 speech features from conformance_test.cc.
_CA01_TRANSCODED_FVNSIM = (
    0.2255266241859742,
    0.7744781572631867,
    0.9054350848438972,
    0.9403641720446017,
    0.9289629387859494,
    0.9260051689927975,
    0.9142438313718589,
    0.919842549719591,
    0.8917748155778098,
    0.8238867650636561,
    0.7984027889339148,
    0.8141862530928756,
    0.8662820989340932,
    0.8668483029767851,
    0.8019144800237413,
    0.8042524050718434,
    0.5669654681801266,
    0.3511008986375132,
    0.13569700558458248,
    0.03456826600725134,
    0.034813587025026384,
)
_CA01_TRANSCODED_FVNSIM10 = (
    -0.03175650288756454,
    0.7578106165755282,
    0.8644200043477909,
    0.9065647347269996,
    0.8405034164117392,
    0.8405737072296843,
    0.8449343263206165,
    0.8288983597361259,
    0.7499022406495988,
    0.6597267054772817,
    0.647420457903445,
    0.6028879600819533,
    0.775422304536276,
    0.7611422464166145,
    0.671803033390387,
    0.6358052170056555,
    0.317992837284678,
    0.04090712874298069,
    0.028282749050483936,
    1.5706238663287824e-05,
    3.757549809964067e-05,
)
_CA01_TRANSCODED_FSTDNSIM = (
    0.3595283888872614,
    0.19298538081528482,
    0.2110829883814835,
    0.149488523104206,
    0.17451187256570547,
    0.17647165905193066,
    0.1674113436740918,
    0.1633609677989578,
    0.1810843383577011,
    0.25536937808682125,
    0.2171265674639381,
    0.25258100178872783,
    0.1715359439654385,
    0.19265217727587086,
    0.237787395596894,
    0.26279954184807924,
    0.43340663882627006,
    0.4706795277642426,
    0.20184770245896538,
    0.14385370015102975,
    0.17621102278928938,
)
_CA01_TRANSCODED_FVDEGENERGY = (
    3.7299720251124406,
    4.470830208309134,
    10.55840038988601,
    16.35949074345499,
    12.800992302256407,
    16.80269599066958,
    16.61333188670338,
    14.74099759914994,
    11.952528154699095,
    11.410803646151837,
    11.623122389330467,
    12.149707991456442,
    11.525571529790318,
    9.84489737132464,
    8.602910980308442,
    8.51688849366786,
    8.272614316812234,
    6.376337211442183,
    0.6417269399396985,
    0.04180321024123891,
    0.0,
)


def _assert_pure_python_value(val: object) -> None:
  assert not isinstance(val, (jax.Array, jax.core.Tracer))
  assert isinstance(val, (int, float, bytes, tuple))
  if isinstance(val, tuple):
    for item in val:
      _assert_pure_python_value(item)


def _assert_pure_python_cached(obj: lattice.LatticeModel) -> None:
  """Assert every stored field is an immutable Python primitive or tuple."""
  assert isinstance(hash(obj), int)
  for field in dataclasses.fields(obj):
    _assert_pure_python_value(getattr(obj, field.name))
  for val in vars(obj).values():
    _assert_pure_python_value(val)
  with pytest.raises(dataclasses.FrozenInstanceError):
    obj._shapes = ()  # type: ignore[misc]  # pylint: disable=protected-access


def _lattice_arrays(
    *,
    num_bands: int = 21,
    const_mos: float = 3.5,
    num_lattices: int = 1,
    lattice_rank: int = 1,
    in_k: int = 1,
    out_k: int = 1,
) -> dict[str, np.ndarray]:
  """Build a minimal valid NumPy dictionary matching `LatticeModel` fields."""
  nf = 4 * num_bands + 1
  out_w = np.zeros((out_k + 1,), dtype=np.float32)
  out_w[0] = np.float32(const_mos)
  return {
      "in_keypoints": np.zeros((nf, in_k), dtype=np.float32),
      "in_inv_lengths": np.ones((nf, in_k), dtype=np.float32),
      "in_has_missing": np.zeros((nf,), dtype=np.bool_),
      "in_missing_value": np.zeros((nf,), dtype=np.float32),
      "lattice_feature_indices": np.zeros(
          (num_lattices, lattice_rank), dtype=np.int32
      ),
      "lattice_calibrator_weights": np.zeros(
          (num_lattices, lattice_rank, in_k + 1), dtype=np.float32
      ),
      "lattice_calibrator_missing_output": np.zeros(
          (num_lattices, lattice_rank), dtype=np.float32
      ),
      "lattice_weights": np.zeros(
          (num_lattices, 1 << lattice_rank), dtype=np.float32
      ),
      "ensemble_weights": np.zeros((num_lattices,), dtype=np.float32),
      "out_keypoints": np.zeros((out_k,), dtype=np.float32),
      "out_inv_lengths": np.ones((out_k,), dtype=np.float32),
      "out_weights": out_w,
  }


def _write_lattice_npz(
    path: pathlib.Path,
    payload: Mapping[str, object],
    *,
    compress: bool = True,
) -> pathlib.Path:
  with path.open("wb") as fh:
    if compress:
      np.savez_compressed(fh, **payload)
    else:
      np.savez(fh, **payload)
  return path


def _py_eval_lattice(lat: lattice.LatticeModel, x85: list[float]) -> float:
  """Reference multilinear lattice evaluation in Python over `lat`."""
  in_kp = lat.in_keypoints.tolist()
  in_inv = lat.in_inv_lengths.tolist()
  in_miss = lat.in_has_missing.tolist()
  in_mval = lat.in_missing_value.tolist()
  fidx = lat.lattice_feature_indices.tolist()
  cal_w = lat.lattice_calibrator_weights.tolist()
  cal_miss = lat.lattice_calibrator_missing_output.tolist()
  lat_w = lat.lattice_weights.tolist()
  ens_w = lat.ensemble_weights.tolist()
  out_kp = lat.out_keypoints.tolist()
  out_inv = lat.out_inv_lengths.tolist()
  out_w = lat.out_weights.tolist()

  basis = [
      [1.0]
      + [
          min(1.0, max(0.0, (x85[f] - in_kp[f][k]) * in_inv[f][k]))
          for k in range(len(in_kp[f]))
      ]
      for f in range(len(x85))
  ]
  z = 0.0
  for l_idx, row_fidx in enumerate(fidx):
    c_row: list[float] = []
    for d, feat in enumerate(row_fidx):
      if in_miss[feat] and x85[feat] == in_mval[feat]:
        c_row.append(cal_miss[l_idx][d])
      else:
        weights = cal_w[l_idx][d]
        c_row.append(
            sum(b * w for b, w in zip(basis[feat], weights, strict=True))
        )
    cur = list(lat_w[l_idx])
    for c_val in c_row:
      w0 = 1.0 - min(abs(c_val), 1.0)
      w1 = 1.0 - min(abs(c_val - 1.0), 1.0)
      half = len(cur) // 2
      cur = [cur[i] * w0 + cur[half + i] * w1 for i in range(half)]
    z += cur[0] * ens_w[l_idx]

  out_basis = [1.0] + [
      min(1.0, max(0.0, (z - out_kp[k]) * out_inv[k]))
      for k in range(len(out_kp))
  ]
  return sum(b * w for b, w in zip(out_basis, out_w, strict=True))


def test_lattice_bundled_npz_and_tflite_rejected(tmp_path: pathlib.Path):
  assert mappers.MODELS.joinpath("speech_lattice.npz").is_file()

  loaded = lattice.load_lattice_model(num_bands=21)
  assert loaded is lattice.load_lattice_model()
  _assert_pure_python_cached(loaded)
  with pytest.raises(ValueError, match="expected 32"):
    lattice.load_lattice_model(num_bands=32)

  fake_tflite = tmp_path / "custom.tflite"
  fake_tflite.write_bytes(b"TFL3-synthetic-custom-model-bytes")
  with pytest.raises(ValueError, match=r"export_lattice_npz\.py"):
    lattice.load_lattice_model(fake_tflite)


def test_predict_lattice_conformance_sentinels_x64_and_jit_grad():
  model = lattice.load_lattice_model()
  assert model.lattice_weights.shape == (60, 4096)
  assert model.lattice_weights.dtype == jnp.float32
  assert model.lattice_feature_indices.dtype == jnp.int32
  assert model.in_has_missing.dtype == jnp.bool_
  with pytest.raises(TypeError):
    model.ensemble_weights[0] = 0.0  # type: ignore[index]

  j_predict = jax.jit(lattice.predict_lattice)

  # High and low observations from tests/tflite_quality_mapper_test.cc.
  ones21 = jnp.ones((21,), dtype=jnp.float32)
  zeros21 = jnp.zeros((21,), dtype=jnp.float32)
  good_mos = j_predict(ones21, ones21, ones21, ones21)
  bad_mos = j_predict(
      jnp.full((21,), 0.2, dtype=jnp.float32),
      jnp.full((21,), 0.1, dtype=jnp.float32),
      ones21,
      ones21,
  )
  assert good_mos.dtype == jnp.float32 and bad_mos.dtype == jnp.float32
  assert 3.0 < float(good_mos) <= 5.0
  assert 1.0 <= float(bad_mos) < 3.0
  assert float(good_mos) == pytest.approx(4.20897102355957, abs=1e-4)
  assert float(bad_mos) == pytest.approx(1.0, abs=1e-6)
  assert float(good_mos) == pytest.approx(
      _py_eval_lattice(model, [1.0] * 84 + [0.5]), abs=1e-5
  )
  assert float(bad_mos) == pytest.approx(
      _py_eval_lattice(model, [0.2] * 21 + [0.1] * 21 + [1.0] * 42 + [0.5]),
      abs=1e-5,
  )

  # CA01_01 identity and transcoded conformance lattice feature vectors.
  ca01_perf = j_predict(ones21, ones21, zeros21, _CA01_PERFECT_DEGENERGY)
  ca01_trans = j_predict(
      _CA01_TRANSCODED_FVNSIM,
      _CA01_TRANSCODED_FVNSIM10,
      _CA01_TRANSCODED_FSTDNSIM,
      _CA01_TRANSCODED_FVDEGENERGY,
  )
  assert float(ca01_perf) == pytest.approx(4.505544185638428, abs=1e-4)
  assert float(ca01_trans) == pytest.approx(3.312926769256592, abs=1e-4)

  # Missing sentinels (fvnsim == 0.7, fvnsim10 == 0.5) select missing_output.
  miss = j_predict(
      jnp.full((21,), 0.7, jnp.float32),
      jnp.full((21,), 0.5, jnp.float32),
      zeros21,
      zeros21,
  )
  near = j_predict(
      jnp.full((21,), 0.7001, jnp.float32),
      jnp.full((21,), 0.5001, jnp.float32),
      zeros21,
      zeros21,
  )
  assert abs(float(miss) - float(near)) > 1e-2
  assert float(miss) == pytest.approx(
      _py_eval_lattice(
          model,
          [float(jnp.float32(0.7))] * 21 + [0.5] * 21 + [0.0] * 42 + [0.5],
      ),
      abs=1e-5,
  )

  # Batched vmap over feature vectors and tau.
  v_tau = jax.jit(jax.vmap(lattice.predict_lattice, in_axes=(0, 0, 0, 0, 0)))(
      jnp.stack([ones21, jnp.asarray(_CA01_TRANSCODED_FVNSIM, jnp.float32)]),
      jnp.stack([ones21, jnp.asarray(_CA01_TRANSCODED_FVNSIM10, jnp.float32)]),
      jnp.stack([zeros21, jnp.asarray(_CA01_TRANSCODED_FSTDNSIM, jnp.float32)]),
      jnp.stack([
          jnp.asarray(_CA01_PERFECT_DEGENERGY, jnp.float32),
          jnp.asarray(_CA01_TRANSCODED_FVDEGENERGY, jnp.float32),
      ]),
      jnp.array([0.5, 0.5], dtype=jnp.float32),
  )
  assert v_tau.shape == (2,) and v_tau.dtype == jnp.float32
  assert float(v_tau[0]) == pytest.approx(float(ca01_perf), abs=1e-5)
  assert float(v_tau[1]) == pytest.approx(float(ca01_trans), abs=1e-5)

  with pytest.raises(ValueError, match="length 21"):
    lattice.predict_lattice([1.0] * 32, [1.0] * 21, [1.0] * 21, [1.0] * 21)
  with pytest.raises(ValueError, match="not bool"):
    lattice.predict_lattice([True] * 21, [1.0] * 21, [1.0] * 21, [1.0] * 21)
  bad_taus: tuple[object, ...] = (
      [0.5],
      True,
      float("nan"),
      float("inf"),
      1.0e100,
      10**400,
      1.0 + 1.0j,
  )
  for bad_tau in bad_taus:
    with pytest.raises(ValueError):
      lattice.predict_lattice(
          [1.0] * 21,
          [1.0] * 21,
          [1.0] * 21,
          [1.0] * 21,
          tau=bad_tau,  # type: ignore[arg-type]
      )

  # Under x64=True, f64 inputs coerce to f32 and f32 HLO has zero f64 ops.
  assert not jax.config.read("jax_enable_x64")
  with jax.enable_x64(True):
    x21_64 = jnp.ones((21,), dtype=jnp.float64)
    assert (
        lattice.predict_lattice(x21_64, x21_64, x21_64, x21_64).dtype
        == jnp.float32
    )
    with pytest.raises(ValueError, match="finite"):
      lattice.predict_lattice(
          ones21, ones21, ones21, ones21, tau=jnp.float64(1.0e100)
      )
    hlo32 = (
        jax.jit(lattice.predict_lattice)
        .lower(ones21, ones21, ones21, ones21)
        .as_text()
    )
    assert "f64" not in hlo32

  # Cold-cache jit + grad leaves cached model pure Python with zero tracers.
  getattr(lattice, "_load_default_lattice_model").cache_clear()
  grad_lat = jax.jit(
      jax.grad(lattice.predict_lattice, argnums=(0, 1, 2, 3, 4))
  )(
      jnp.full((21,), 0.8, dtype=jnp.float32),
      jnp.full((21,), 0.7, dtype=jnp.float32),
      jnp.full((21,), 0.1, dtype=jnp.float32),
      jnp.full((21,), 8.0, dtype=jnp.float32),
      jnp.float32(0.5),
  )
  assert all(bool(jnp.all(jnp.isfinite(g))) for g in grad_lat)
  _assert_pure_python_cached(lattice.load_lattice_model())


def test_custom_lattice_npz_loading_reload_and_validation(
    tmp_path: pathlib.Path,
):
  custom_npz = _write_lattice_npz(
      tmp_path / "custom_lattice.npz",
      _lattice_arrays(num_bands=4, const_mos=2.75),
      compress=False,
  )
  loaded = lattice.load_lattice_model(custom_npz, num_bands=4)
  assert loaded is lattice.load_lattice_model(custom_npz, num_bands=4)
  _assert_pure_python_cached(loaded)
  assert loaded.in_keypoints.shape == (17, 1)

  z4 = jnp.zeros((4,), dtype=jnp.float32)
  pred = lattice.predict_lattice(z4, z4, z4, z4, svr_model_path=custom_npz)
  assert pred.dtype == jnp.float32
  assert float(pred) == pytest.approx(2.75, abs=1e-6)
  pred_model = lattice.predict_lattice(z4, z4, z4, z4, model=loaded)
  assert pred_model.dtype == jnp.float32
  assert float(pred_model) == pytest.approx(2.75, abs=1e-6)
  with pytest.raises(ValueError, match="at most one"):
    lattice.predict_lattice(
        z4, z4, z4, z4, model=loaded, svr_model_path=custom_npz
    )
  with pytest.raises(TypeError, match="LatticeModel"):
    lattice.predict_lattice(
        z4, z4, z4, z4, model="bad"  # type: ignore[arg-type]
    )

  # Compressed .npz loading, mtime cache invalidation, and CAS blob symlinks.
  custom_cmp = _write_lattice_npz(
      tmp_path / "custom_compressed.npz",
      _lattice_arrays(num_bands=4, const_mos=3.125),
      compress=True,
  )
  assert float(
      lattice.predict_lattice(z4, z4, z4, z4, svr_model_path=custom_cmp)
  ) == pytest.approx(3.125, abs=1e-6)

  prev_mtime = custom_cmp.stat().st_mtime_ns
  _write_lattice_npz(
      custom_cmp, _lattice_arrays(num_bands=4, const_mos=4.125), compress=True
  )
  os.utime(custom_cmp, ns=(prev_mtime + 1_000_000, prev_mtime + 1_000_000))
  assert float(
      lattice.predict_lattice(z4, z4, z4, z4, svr_model_path=custom_cmp)
  ) == pytest.approx(4.125, abs=1e-6)

  cas_lat = tmp_path / "lat_cas_blob"
  cas_lat.write_bytes(custom_cmp.read_bytes())
  lat_link = tmp_path / "linked_lat.npz"
  lat_link.symlink_to(cas_lat)
  assert float(
      lattice.predict_lattice(z4, z4, z4, z4, svr_model_path=lat_link)
  ) == pytest.approx(4.125, abs=1e-6)
  suffixless_lat_link = tmp_path / "lat_link_no_ext"
  suffixless_lat_link.symlink_to(custom_cmp)
  assert float(
      lattice.predict_lattice(
          z4, z4, z4, z4, svr_model_path=suffixless_lat_link
      )
  ) == pytest.approx(4.125, abs=1e-6)

  # Cold-cache jit over custom .npz and atomic same-mtime/size os.replace.
  getattr(lattice, "_load_custom_lattice_model").cache_clear()
  jit_custom_lat = jax.jit(
      lambda v: lattice.predict_lattice(v, v, v, v, svr_model_path=custom_cmp)
  )(z4)
  assert float(jit_custom_lat) == pytest.approx(4.125, abs=1e-6)
  _assert_pure_python_cached(lattice.load_lattice_model(custom_cmp))

  lat_stat = custom_npz.stat()
  staged_lat = _write_lattice_npz(
      tmp_path / "staged_lat.npz",
      _lattice_arrays(num_bands=4, const_mos=1.75),
      compress=False,
  )
  os.utime(staged_lat, ns=(lat_stat.st_mtime_ns, lat_stat.st_mtime_ns))
  os.replace(staged_lat, custom_npz)
  assert float(
      lattice.predict_lattice(z4, z4, z4, z4, svr_model_path=custom_npz)
  ) == pytest.approx(1.75, abs=1e-6)

  # Input array mutation after LatticeModel construction cannot corrupt weights.
  mut_payload = _lattice_arrays(num_bands=2, const_mos=3.25)
  mut_out_w = mut_payload["out_weights"]
  unaliased_lat = lattice.LatticeModel(**mut_payload)
  mut_out_w[0] = np.float32("nan")
  _assert_pure_python_cached(unaliased_lat)
  assert float(unaliased_lat.out_weights[0]) == 3.25

  with pytest.raises(ValueError, match="21 bands"):
    lattice.load_lattice_model(custom_cmp, num_bands=21)
  with pytest.raises(ValueError, match="expected 5"):
    lattice.load_lattice_model(num_bands=5)
  for bad_path in ("", "bad\x00path", b"/tmp/a.npz", bytearray(b"/tmp/a"), 123):
    with pytest.raises(ValueError, match="non-empty"):
      lattice.load_lattice_model(bad_path)  # type: ignore[arg-type]

  bad_cases: list[tuple[str, dict[str, object], str]] = [
      (
          "missing_key",
          {
              k: v
              for k, v in _lattice_arrays(num_bands=2).items()
              if k != "out_weights"
          },
          "missing required keys",
      ),
      (
          "unexpected_key",
          {**_lattice_arrays(num_bands=2), "extra_unknown": np.array([1])},
          "unexpected keys",
      ),
      (
          "too_few_features",
          {
              **_lattice_arrays(num_bands=1),
              "in_keypoints": np.zeros((1, 1), dtype=np.float32),
          },
          ">=5 features",
      ),
      (
          "bad_feature_count",
          {
              **_lattice_arrays(num_bands=2),
              "in_keypoints": np.zeros((10, 1), dtype=np.float32),
          },
          r"4 \* num_bands \+ 1",
      ),
      (
          "bad_idx_range",
          {
              **_lattice_arrays(num_bands=2),
              "lattice_feature_indices": np.array([[9]], dtype=np.int32),
          },
          r"must be in \[0, 9\)",
      ),
      (
          "bad_idx_neg",
          {
              **_lattice_arrays(num_bands=2),
              "lattice_feature_indices": np.array([[-1]], dtype=np.int32),
          },
          r"must be in \[0, 9\)",
      ),
      (
          "bad_idx_int64_wrap",
          {
              **_lattice_arrays(num_bands=2),
              "lattice_feature_indices": np.array([[1 << 32]], dtype=np.int64),
          },
          r"must be in \[0, 9\)",
      ),
      (
          "bad_idx_float",
          {
              **_lattice_arrays(num_bands=2),
              "lattice_feature_indices": np.array([[0.0]], dtype=np.float32),
          },
          "integer dtype",
      ),
      (
          "bad_idx_bool",
          {
              **_lattice_arrays(num_bands=2),
              "lattice_feature_indices": np.array([[True]], dtype=np.bool_),
          },
          "integer dtype",
      ),
      (
          "bad_missing_dtype",
          {
              **_lattice_arrays(num_bands=2),
              "in_has_missing": np.zeros((9,), dtype=np.int32),
          },
          "bool dtype",
      ),
      (
          "bad_weight_bool",
          {
              **_lattice_arrays(num_bands=2),
              "out_weights": np.array([True, False], dtype=np.bool_),
          },
          "floating dtype",
      ),
      (
          "bad_weight_str",
          {
              **_lattice_arrays(num_bands=2),
              "out_weights": np.array(["3.0", "0.0"]),
          },
          "floating dtype",
      ),
      (
          "nonfinite_weight",
          {
              **_lattice_arrays(num_bands=2),
              "out_weights": np.array([np.nan, 0.0], dtype=np.float32),
          },
          "finite",
      ),
      (
          "inf_weight",
          {
              **_lattice_arrays(num_bands=2),
              "out_weights": np.array([np.inf, 0.0], dtype=np.float32),
          },
          "finite",
      ),
      (
          "f64_overflow_to_f32_weight",
          {
              **_lattice_arrays(num_bands=2),
              "out_weights": np.array([1.0e100, 0.0], dtype=np.float64),
          },
          "finite",
      ),
      (
          "lattice_rank_shift_overflow",
          {
              **_lattice_arrays(num_bands=2, lattice_rank=1),
              "lattice_feature_indices": np.zeros((1, 65), dtype=np.int32),
              "lattice_calibrator_weights": np.zeros(
                  (1, 65, 2), dtype=np.float32
              ),
              "lattice_calibrator_missing_output": np.zeros(
                  (1, 65), dtype=np.float32
              ),
              "lattice_weights": np.zeros((1, 2), dtype=np.float32),
          },
          r"2\*\*65",
      ),
      (
          "bad_lattice_weights_shape",
          {
              **_lattice_arrays(num_bands=2, lattice_rank=2),
              "lattice_weights": np.zeros((1, 3), dtype=np.float32),
          },
          r"2\*\*2",
      ),
      (
          "bad_cal_weights_shape",
          {
              **_lattice_arrays(num_bands=2, in_k=3),
              "lattice_calibrator_weights": np.zeros(
                  (1, 1, 2), dtype=np.float32
              ),
          },
          "lattice_calibrator_weights",
      ),
      (
          "bad_out_weights_shape",
          {
              **_lattice_arrays(num_bands=2, out_k=2),
              "out_weights": np.zeros((2,), dtype=np.float32),
          },
          "out_weights",
      ),
  ]
  for name, payload, pattern in bad_cases:
    bad_path = _write_lattice_npz(tmp_path / f"{name}.npz", payload)
    with pytest.raises(ValueError, match=pattern):
      lattice.load_lattice_model(bad_path)

  # Direct LatticeModel construction also rejects non-float, bool, ragged, NaN.
  int_ow = _lattice_arrays(num_bands=2)
  with pytest.raises(ValueError, match="floating dtype"):
    lattice.LatticeModel(
        **{**int_ow, "out_weights": jnp.array([3, 0], dtype=jnp.int32)}
    )
  with pytest.raises(ValueError, match="floating dtype"):
    lattice.LatticeModel(**{**int_ow, "out_weights": [True, 0.0]})
  with pytest.raises(ValueError, match="ragged"):
    lattice.LatticeModel(
        **{**int_ow, "in_keypoints": [[0.0]] * 8 + [[0.0, 1.0]]}
    )
  with pytest.raises(ValueError, match="finite"):
    lattice.LatticeModel(**{**int_ow, "out_weights": [float("nan"), 0.0]})
  with pytest.raises(ValueError, match="finite"):
    lattice.LatticeModel(**{**int_ow, "out_weights": [10**40, 0.0]})

  # Pickled object arrays inside .npz must be rejected (allow_pickle=False).
  pickled_npz = _write_lattice_npz(
      tmp_path / "pickled.npz",
      {
          **_lattice_arrays(num_bands=2),
          "out_weights": np.array([{"a": 1}, None], dtype=object),
      },
  )
  with pytest.raises(ValueError, match=r"Corrupt \.npz"):
    lattice.load_lattice_model(pickled_npz)

  # Plain .npy (including forged huge shape) renamed to .npz must be rejected.
  npy_as_npz = tmp_path / "plain_array.npz"
  with npy_as_npz.open("wb") as fh:
    np.save(fh, np.zeros((3,), dtype=np.float32))
  with pytest.raises(ValueError, match=r"\.npz archive"):
    lattice.load_lattice_model(npy_as_npz)

  def _make_npy_bytes(
      shape: tuple[int, ...],
      descr: object,
      *,
      version: tuple[int, int] = (1, 0),
      payload: bytes = b"",
  ) -> bytes:
    hdr = (
        f"{{'descr': {descr!r}, 'fortran_order': False,"
        f" 'shape': {shape!r}, }}\n"
    )
    hdr_bytes = hdr.encode("utf-8" if version == (3, 0) else "latin1")
    len_fmt = "<H" if version == (1, 0) else "<I"
    return (
        np.lib.format.magic(*version)
        + struct.pack(len_fmt, len(hdr_bytes))
        + hdr_bytes
        + payload
    )

  forged_plain_npy = tmp_path / "forged_plain.npz"
  forged_plain_npy.write_bytes(_make_npy_bytes((1_000_000_000,), "<f4"))
  with pytest.raises(ValueError, match=r"\.npz archive"):
    lattice.load_lattice_model(forged_plain_npy)

  # Forged .npy headers inside a tiny .npz must be rejected before allocation.
  valid_arrays = _lattice_arrays(num_bands=2, const_mos=3.25)

  def _write_npz_with_raw_out_weights(
      dest: pathlib.Path, out_weights_bytes: bytes
  ) -> pathlib.Path:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
      for key, arr in valid_arrays.items():
        if key == "out_weights":
          zf.writestr(f"{key}.npy", out_weights_bytes)
        else:
          buf = io.BytesIO()
          np.save(buf, arr, allow_pickle=False)
          zf.writestr(f"{key}.npy", buf.getvalue())
    return dest

  for bad_name, forged_hdr in (
      ("forged_v1.npz", _make_npy_bytes((1_000_000_000,), "<f4")),
      (
          "forged_v3.npz",
          _make_npy_bytes((1_000_000_000,), "<f4", version=(3, 0)),
      ),
      ("forged_zero_item.npz", _make_npy_bytes((1_000_000_000,), "|S0")),
      ("forged_overflow_shape.npz", _make_npy_bytes((1 << 64, 0), "<f4")),
      (
          "forged_overflow_descr.npz",
          _make_npy_bytes((1,), ("<f4", (1 << 64,))),
      ),
  ):
    forged_npz = _write_npz_with_raw_out_weights(
        tmp_path / bad_name, forged_hdr
    )
    with pytest.raises(ValueError, match=r"Corrupt \.npz.*out_weights"):
      lattice.load_lattice_model(forged_npz)

  # Valid .npy v3.0 header inside .npz loads cleanly.
  v3_npz = _write_npz_with_raw_out_weights(
      tmp_path / "valid_v3.npz",
      _make_npy_bytes(
          (2,),
          "<f4",
          version=(3, 0),
          payload=valid_arrays["out_weights"].tobytes(),
      ),
  )
  assert float(lattice.load_lattice_model(v3_npz).out_weights[0]) == 3.25

  for bad_npz_name, bad_npz_bytes in (
      ("corrupt.npz", b"PK\x03\x04corrupt-zip-stream"),
      ("empty.npz", b""),
      ("trunc.npz", custom_cmp.read_bytes()[:-10]),
  ):
    bad_lat_npz = tmp_path / bad_npz_name
    bad_lat_npz.write_bytes(bad_npz_bytes)
    with pytest.raises(ValueError, match=r"Corrupt \.npz lattice model"):
      lattice.load_lattice_model(bad_lat_npz)

  bomb_npz = tmp_path / "bomb.npz"
  with zipfile.ZipFile(
      bomb_npz, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
  ) as zf:
    for key in _lattice_arrays(num_bands=2):
      payload_bytes = (
          b"\x00" * (64 * 1024 * 1024 + 1) if key == "out_weights" else b""
      )
      zf.writestr(f"{key}.npy", payload_bytes)
  with pytest.raises(ValueError, match="uncompressed size exceeds"):
    lattice.load_lattice_model(bomb_npz)

  oversized_npz = tmp_path / "oversized.npz"
  with oversized_npz.open("wb") as fh:
    fh.truncate(64 * 1024 * 1024 + 1)
  with pytest.raises(ValueError, match="byte limit"):
    lattice.load_lattice_model(oversized_npz)

  fifo_npz = tmp_path / "lattice_pipe.npz"
  os.mkfifo(fifo_npz)
  for non_regular in (tmp_path, pathlib.Path("/dev/null"), fifo_npz):
    with pytest.raises(ValueError, match="regular file"):
      lattice.load_lattice_model(non_regular)

  for ext in (".txt", ".bin", ".json", ".json.gz"):
    bad_ext = tmp_path / f"model{ext}"
    bad_ext.write_bytes(b"not a lattice\n")
    with pytest.raises(ValueError, match="Unsupported lattice model format"):
      lattice.load_lattice_model(bad_ext)
