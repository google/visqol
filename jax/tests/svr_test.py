"""Tests for svr.py audio nu-SVR mapper against C++ ViSQOL v3.3.3."""

import dataclasses
import gzip
import math
import os
import pathlib

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import mappers
from visqol_jax import svr

_SVR_TXT_NAME = "libsvm_nu_svr_model.txt"
_CPP_TEST_SVR_REL = "testdata/test_model/cpp_model.txt"

# Contrabassoon 48 kHz 24 kbps AAC FVNSIM (tests/svr_model_training_test.cc).
_CONTRABASSOON_FVNSIM = (
    0.853862,
    0.680331,
    0.535649,
    0.639760,
    0.029999,
    0.058591,
    0.077462,
    0.012432,
    0.192035,
    0.389230,
    0.479403,
    0.419914,
    0.521414,
    0.858340,
    0.884218,
    0.864682,
    0.868514,
    0.845271,
    0.850559,
    0.877882,
    0.903985,
    0.887572,
    0.920558,
    0.920375,
    0.954934,
    0.945048,
    0.952716,
    0.986600,
    0.987345,
    0.936462,
    0.856010,
    0.829761,
)


def _find_repo_file(rel_path: str) -> pathlib.Path:
  """Locate `rel_path` across Blaze runfiles and OSS repo layouts."""
  this_file = pathlib.Path(__file__)
  for base in (this_file, this_file.resolve()):
    for parent in base.parents:
      candidate = parent / rel_path
      if candidate.is_file():
        return candidate
  for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
    env_val = os.environ.get(env_name)
    if env_val:
      root = pathlib.Path(env_val)
      if root.is_dir():
        for match in root.rglob(rel_path):
          if match.is_file():
            return match
  raise FileNotFoundError(f"Could not locate {rel_path}")


def _assert_pure_python_value(val: object) -> None:
  assert not isinstance(val, (jax.Array, jax.core.Tracer))
  assert isinstance(val, (int, float, bytes, tuple))
  if isinstance(val, tuple):
    for item in val:
      _assert_pure_python_value(item)


def _assert_pure_python_cached(obj: svr.SvrModel) -> None:
  """Assert every stored field is an immutable Python primitive or tuple."""
  assert isinstance(hash(obj), int)
  for field in dataclasses.fields(obj):
    _assert_pure_python_value(getattr(obj, field.name))
  for val in vars(obj).values():
    _assert_pure_python_value(val)
  with pytest.raises(dataclasses.FrozenInstanceError):
    obj.gamma = 0.0  # type: ignore[misc]


def _write_libsvm_txt(
    path: pathlib.Path,
    *,
    gamma: float,
    rho: float,
    sv_coef: list[float],
    support_vectors: list[list[tuple[int, float]]],
    svm_type: str = "nu_svr",
    kernel_type: str = "rbf",
    compress: bool = False,
) -> pathlib.Path:
  """Write a synthetic `.txt` or `.txt.gz` libsvm SVR model."""
  lines = [
      f"svm_type {svm_type}",
      f"kernel_type {kernel_type}",
      f"gamma {gamma!r}",
      "nr_class 2",
      f"total_sv {len(sv_coef)}",
      f"rho {rho!r}",
      "",
      "SV",
  ]
  for coef, pairs in zip(sv_coef, support_vectors, strict=True):
    feats = " ".join(f"{idx}:{val!r}" for idx, val in pairs)
    lines.append(f"{coef!r} {feats}".rstrip())
  raw = ("\n".join(lines) + "\n").encode("utf-8")
  path.write_bytes(
      gzip.compress(raw, compresslevel=9, mtime=0) if compress else raw
  )
  return path


def test_bundled_svr_matches_upstream_libsvm_and_legacy_cpp_model():
  assert mappers.MODELS.joinpath("svr_model.txt.gz").is_file()

  txt_path = _find_repo_file(f"model/{_SVR_TXT_NAME}")
  bundled = svr.load_svr_model()
  assert bundled is svr.load_svr_model(num_features=32)
  _assert_pure_python_cached(bundled)
  assert bundled.support_vectors.shape == (317, 32)
  assert bundled.support_vectors.dtype == jnp.float32
  assert bundled.sv_coef.shape == (317,)
  assert bundled.sv_coef.dtype == jnp.float32
  with pytest.raises(TypeError):
    bundled.sv_coef[0] = 0.0  # type: ignore[index]

  parsed = svr.parse_libsvm_svr(txt_path)
  _assert_pure_python_cached(parsed)
  assert bundled == parsed
  assert bool(jnp.array_equal(bundled.sv_coef, parsed.sv_coef))
  assert bool(jnp.array_equal(bundled.support_vectors, parsed.support_vectors))

  # Check parity on Castanets identity conformance and Contrabassoon sample.
  ones = jnp.ones((32,), dtype=jnp.float32)
  pred_ones = svr.predict_svr(ones)
  pred_txt = svr.predict_svr(ones, svr_model_path=txt_path)
  assert pred_ones.dtype == jnp.float32
  assert float(pred_ones) == float(pred_txt)
  assert float(pred_ones) == pytest.approx(4.732101253042348, abs=2e-5)

  pred_cb = svr.predict_svr(_CONTRABASSOON_FVNSIM)
  assert pred_cb.dtype == jnp.float32
  assert float(pred_cb) == pytest.approx(4.305334699145016, abs=2e-5)

  # Legacy 0-indexed testdata/test_model/cpp_model.txt matches C++ k_function.
  cpp_txt = _find_repo_file(_CPP_TEST_SVR_REL)
  cpp_svr = svr.load_svr_model(cpp_txt, num_features=32)
  _assert_pure_python_cached(cpp_svr)
  assert cpp_svr.support_vectors.shape == (79, 32)
  assert bool(jnp.all(cpp_svr.support_vectors[:, 31] == 0.0))

  # Manually evaluate C++ sparse k_function on cpp_model.txt at x == ones(32).
  cpp_lines = cpp_txt.read_text(encoding="utf-8").splitlines()
  sv_start = cpp_lines.index("SV") + 1
  manual = 0.0
  for row_line in cpp_lines[sv_start:]:
    toks = row_line.split()
    coef = float(toks[0])
    kv = {int(t.split(":")[0]): float(t.split(":")[1]) for t in toks[1:]}
    sq = kv.get(0, 0.0) ** 2 + sum(
        (1.0 - kv.get(idx, 0.0)) ** 2 for idx in range(1, 33)
    )
    manual += coef * math.exp(-0.03125 * sq)
  want_cpp = min(5.0, max(1.0, manual - (-3.32948)))
  got_cpp = svr.predict_svr(ones, svr_model_path=cpp_txt)
  assert float(got_cpp) == pytest.approx(want_cpp, abs=1e-6)


def test_svr_custom_txt_and_txt_gz_reload_symlinks_and_validation(
    tmp_path: pathlib.Path,
):
  # Test epsilon_svr, omitted middle feature, and trailing zero padding.
  custom_txt = _write_libsvm_txt(
      tmp_path / "custom.txt",
      gamma=0.25,
      rho=-2.5,
      sv_coef=[0.6, -0.2],
      support_vectors=[[(1, 1.0), (3, 0.25)], [(2, 0.5)]],
      svm_type="epsilon_svr",
  )
  parsed = svr.load_svr_model(custom_txt, num_features=4)
  _assert_pure_python_cached(parsed)
  assert parsed.support_vectors.shape == (2, 4)
  assert parsed.support_vectors.tolist() == [
      [1.0, 0.0, 0.25, 0.0],
      [0.0, 0.5, 0.0, 0.0],
  ]

  x = [0.8, 0.5, 0.25, 0.0]
  expected = (
      0.6 * math.exp(-0.25 * (0.2**2 + 0.5**2))
      - 0.2 * math.exp(-0.25 * (0.8**2 + 0.25**2))
      + 2.5
  )
  got = svr.predict_svr(x, svr_model_path=custom_txt)
  assert got.dtype == jnp.float32
  assert float(got) == pytest.approx(expected, abs=1e-6)

  # All-zero support vector pads when num_features is set, fails when None.
  zero_sv_txt = _write_libsvm_txt(
      tmp_path / "zero_sv.txt",
      gamma=0.1,
      rho=-2.0,
      sv_coef=[0.5],
      support_vectors=[[]],
  )
  zero_svr = svr.parse_libsvm_svr(zero_sv_txt, num_features=3)
  assert zero_svr.support_vectors.tolist() == [[0.0, 0.0, 0.0]]
  with pytest.raises(ValueError, match="no feature indices"):
    svr.parse_libsvm_svr(zero_sv_txt)
  with pytest.raises(ValueError, match="exceeds"):
    svr.load_svr_model(custom_txt, num_features=2)

  # Custom .txt.gz loading, mtime cache invalidation, and CAS blob symlinks.
  custom_gz = _write_libsvm_txt(
      tmp_path / "custom_svr.txt.gz",
      gamma=0.25,
      rho=-2.5,
      sv_coef=[0.6, -0.2],
      support_vectors=[[(1, 1.0), (3, 0.25)], [(2, 0.5)]],
      compress=True,
  )
  assert float(svr.predict_svr(x, svr_model_path=custom_gz)) == pytest.approx(
      expected, abs=1e-6
  )
  prev_mtime = custom_gz.stat().st_mtime_ns
  _write_libsvm_txt(
      custom_gz,
      gamma=0.25,
      rho=-3.0,
      sv_coef=[0.6, -0.2],
      support_vectors=[[(1, 1.0), (3, 0.25)], [(2, 0.5)]],
      compress=True,
  )
  os.utime(custom_gz, ns=(prev_mtime + 1_000_000, prev_mtime + 1_000_000))
  assert float(svr.predict_svr(x, svr_model_path=custom_gz)) == pytest.approx(
      expected + 0.5, abs=1e-6
  )

  cas_svr = tmp_path / "svr_cas_blob"
  cas_svr.write_bytes(custom_gz.read_bytes())
  svr_link = tmp_path / "linked_svr.txt.gz"
  svr_link.symlink_to(cas_svr)
  assert float(svr.predict_svr(x, svr_model_path=svr_link)) == pytest.approx(
      expected + 0.5, abs=1e-6
  )
  assert float(svr.predict_svr(x, svr_model_path=cas_svr)) == pytest.approx(
      expected + 0.5, abs=1e-6
  )

  # Clamping to [1, 5], list mutation immunity, and jit + vmap.
  mut_sv = [[1.0, 1.0, 1.0, 1.0]]
  hi = svr.SvrModel(0.01, -10.0, [1.0], mut_sv)
  lo = svr.SvrModel(0.01, 10.0, [1.0], [[1.0, 1.0, 1.0, 1.0]])
  mut_sv[0][0] = float("nan")
  _assert_pure_python_cached(hi)
  assert float(hi.support_vectors[0, 0]) == 1.0
  assert float(svr.predict_svr([1.0, 1.0, 1.0, 1.0], model=hi)) == 5.0
  assert float(svr.predict_svr([1.0, 1.0, 1.0, 1.0], model=lo)) == 1.0
  int_svr = svr.SvrModel(
      0.01, 0.0, jnp.array([1], jnp.int32), jnp.array([[1, 2]], jnp.int32)
  )
  assert int_svr.num_sv == 1 and int_svr.num_features == 2

  batch = jnp.stack([
      jnp.ones(32, jnp.float32),
      jnp.asarray(_CONTRABASSOON_FVNSIM, dtype=jnp.float32),
  ])
  batched = jax.jit(jax.vmap(svr.predict_svr))(batch)
  assert batched.shape == (2,) and batched.dtype == jnp.float32
  assert float(batched[0]) == pytest.approx(4.732101253042348, abs=2e-5)
  assert float(batched[1]) == pytest.approx(4.305334699145016, abs=2e-5)
  _assert_pure_python_cached(svr.load_svr_model())

  # Under x64=True, f64 inputs coerce to f32 and f32 HLO has zero f64 ops.
  assert not jax.config.read("jax_enable_x64")
  with jax.enable_x64(True):
    x32_64 = jnp.ones((32,), dtype=jnp.float64)
    assert svr.predict_svr(x32_64).dtype == jnp.float32
    x32_32 = jnp.ones((32,), dtype=jnp.float32)
    hlo32 = jax.jit(svr.predict_svr).lower(x32_32).as_text()
    assert "f64" not in hlo32

  # Cold-cache jit + grad leaves cached model pure Python with zero tracers.
  getattr(svr, "_load_default_svr_model").cache_clear()
  grad_svr = jax.jit(jax.grad(svr.predict_svr))(
      jnp.full((32,), 0.8, dtype=jnp.float32)
  )
  assert bool(jnp.all(jnp.isfinite(grad_svr)))
  _assert_pure_python_cached(svr.load_svr_model())

  # Validation errors on malformed libsvm files, corrupt gzip, and arguments.
  hdr = "svm_type nu_svr\nkernel_type rbf\ngamma 0.1\ntotal_sv 1\nrho 0\n"
  bad_txt_cases: list[tuple[str, str, str]] = [
      (
          "bad_svm_type",
          "svm_type c_svc\nkernel_type rbf\ngamma 0.1\ntotal_sv 1\nrho 0\n"
          + "SV\n1 1:1\n",
          "svm_type",
      ),
      (
          "bad_kernel",
          "svm_type nu_svr\nkernel_type linear\ngamma 0.1\ntotal_sv 1\nrho 0\n"
          + "SV\n1 1:1\n",
          "kernel_type",
      ),
      (
          "missing_rho",
          "svm_type nu_svr\nkernel_type rbf\ngamma 0.1\ntotal_sv 1\n"
          + "SV\n1 1:1\n",
          "Missing required",
      ),
      ("missing_sv_sec", hdr, "Missing SV"),
      (
          "zero_total_sv",
          "svm_type nu_svr\nkernel_type rbf\ngamma 0.1\ntotal_sv 0\nrho 0\n"
          + "SV\n",
          "positive total_sv",
      ),
      (
          "wrong_sv_count",
          "svm_type nu_svr\nkernel_type rbf\ngamma 0.1\ntotal_sv 2\nrho 0\n"
          + "SV\n1 1:1\n",
          "Expected 2",
      ),
      (
          "bad_header",
          "svm_type nu_svr\nbrokenheaderline\n",
          "Malformed libsvm header",
      ),
      ("bad_token", hdr + "SV\n1 nocolon\n", "Malformed libsvm feature"),
      ("neg_idx", hdr + "SV\n1 -1:1.0\n", "non-negative"),
      ("zero_only_idx", hdr + "SV\n1 0:1.0\n", "no feature indices"),
      ("unsorted_idx", hdr + "SV\n1 2:1.0 2:0.5\n", "strictly increasing"),
      ("nan_val", hdr + "SV\n1 1:nan\n", "Non-finite"),
      ("f32_overflow_val", hdr + "SV\n1 1:1e100\n", "finite"),
      ("f32_overflow_idx0", hdr + "SV\n1 0:1e100 1:1.0\n", "finite"),
      (
          "f32_overflow_coef_masked",
          "svm_type nu_svr\nkernel_type rbf\ngamma 1.0\ntotal_sv 1\nrho 0\n"
          + "SV\n1e100 0:20.0 1:1.0\n",
          "finite",
      ),
      ("large_int_coef", hdr + f"SV\n{10**40} 1:1.0\n", "finite"),
      (
          "neg_gamma",
          "svm_type nu_svr\nkernel_type rbf\ngamma -0.1\ntotal_sv 1\nrho 0\n"
          + "SV\n1 1:1\n",
          "non-negative",
      ),
      (
          "overflow_gamma",
          "svm_type nu_svr\nkernel_type rbf\ngamma 1e100\ntotal_sv 1\nrho 0\n"
          + "SV\n1 1:1\n",
          "finite",
      ),
      (
          "bad_nr_class",
          "svm_type nu_svr\nkernel_type rbf\ngamma 0.1\nnr_class 3\n"
          + "total_sv 1\nrho 0\nSV\n1 1:1\n",
          "nr_class 2",
      ),
      (
          "huge_feat_idx",
          hdr + "SV\n1.0 1000000000:1.0\n",
          "exceeds limit",
      ),
      (
          "huge_sv_matrix",
          "svm_type nu_svr\nkernel_type rbf\ngamma 0.1\ntotal_sv 300\nrho 0\n"
          + "SV\n"
          + "1.0 60000:1.0\n" * 300,
          "matrix size.*exceeds limit",
      ),
  ]
  for name, content, pattern in bad_txt_cases:
    p = tmp_path / f"{name}.txt"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
      svr.parse_libsvm_svr(p)

  for bad_gz_name, bad_gz_bytes in (
      ("corrupt_svr.txt.gz", b"\x1f\x8b\x08\x00corrupt-gzip-stream"),
      ("empty_svr.txt.gz", b""),
      ("trunc_svr.txt.gz", custom_gz.read_bytes()[:-5]),
  ):
    bad_gz = tmp_path / bad_gz_name
    bad_gz.write_bytes(bad_gz_bytes)
    with pytest.raises(ValueError, match="Corrupt gzip SVR model"):
      svr.load_svr_model(bad_gz)

  bomb_gz = tmp_path / "bomb_svr.txt.gz"
  bomb_gz.write_bytes(
      gzip.compress(b" " * (64 * 1024 * 1024 + 1), compresslevel=1, mtime=0)
  )
  with pytest.raises(ValueError, match="byte limit"):
    svr.load_svr_model(bomb_gz)

  oversized_txt = tmp_path / "oversized_svr.txt"
  with oversized_txt.open("wb") as fh:
    fh.truncate(64 * 1024 * 1024 + 1)
  with pytest.raises(ValueError, match="byte limit"):
    svr.load_svr_model(oversized_txt)

  fifo_svr = tmp_path / "svr_pipe.txt"
  os.mkfifo(fifo_svr)
  for non_regular in (tmp_path, pathlib.Path("/dev/null"), fifo_svr):
    with pytest.raises(ValueError, match="regular file"):
      svr.load_svr_model(non_regular)

  # Cold-cache jit over custom .txt.gz and atomic same-mtime/size os.replace.
  jit_svr_gz = _write_libsvm_txt(
      tmp_path / "jit_svr.txt.gz",
      gamma=0.0,
      rho=-2.25,
      sv_coef=[0.0],
      support_vectors=[[(1, 0.0), (4, 0.0)]],
      compress=True,
  )
  getattr(svr, "_load_custom_svr_model").cache_clear()
  jit_custom_svr = jax.jit(
      lambda v: svr.predict_svr(v, svr_model_path=jit_svr_gz)
  )(jnp.asarray(x, dtype=jnp.float32))
  assert float(jit_custom_svr) == pytest.approx(2.25, abs=1e-6)
  _assert_pure_python_cached(svr.load_svr_model(jit_svr_gz))

  same_stat = jit_svr_gz.stat()
  staged_svr = _write_libsvm_txt(
      tmp_path / "staged_svr.txt.gz",
      gamma=0.0,
      rho=-3.25,
      sv_coef=[0.0],
      support_vectors=[[(1, 0.0), (4, 0.0)]],
      compress=True,
  )
  os.utime(staged_svr, ns=(same_stat.st_mtime_ns, same_stat.st_mtime_ns))
  os.replace(staged_svr, jit_svr_gz)
  assert float(svr.predict_svr(x, svr_model_path=jit_svr_gz)) == pytest.approx(
      3.25, abs=1e-6
  )

  bin_svr = tmp_path / "binary_model.tflite"
  bin_svr.write_bytes(b"\xff\xfe\x00TFL3-binary")
  with pytest.raises(ValueError, match="UTF-8"):
    svr.load_svr_model(bin_svr)

  with pytest.raises(ValueError, match="non-negative"):
    svr.SvrModel(-0.1, 0.0, [1.0], [[1.0, 1.0, 1.0, 1.0]])
  with pytest.raises(ValueError, match="sv_coef"):
    svr.SvrModel(0.1, 0.0, [1.0, 1.0], [[1.0, 1.0, 1.0, 1.0]])
  with pytest.raises(ValueError, match="support_vectors"):
    svr.SvrModel(0.1, 0.0, [1.0, 1.0], [1.0, 1.0])  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="ragged"):
    svr.SvrModel(0.1, 0.0, [1.0, 1.0], [[1.0, 2.0], [1.0]])
  with pytest.raises(ValueError, match="expected 21"):
    svr.load_svr_model(num_features=21)
  for bad_path in ("", "bad\x00path", b"/tmp/a.txt", bytearray(b"/tmp/a"), 123):
    with pytest.raises(ValueError, match="non-empty"):
      svr.load_svr_model(bad_path)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="length 32"):
    svr.predict_svr([1.0] * 21)
  with pytest.raises(ValueError, match="non-empty 1D"):
    svr.predict_svr([])
  with pytest.raises(ValueError, match="not bool"):
    svr.predict_svr([True] * 32)
  with pytest.raises(ValueError, match="at most one"):
    svr.predict_svr(x, model=hi, svr_model_path=custom_txt)
  with pytest.raises(TypeError, match="SvrModel"):
    svr.predict_svr(x, model="bad")  # type: ignore[arg-type]
