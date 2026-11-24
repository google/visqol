"""Tests for mappers.py speech exponential mapper against C++ ViSQOL v3.3.3."""

import os
import pathlib

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import _check
from visqol_jax import mappers


def test_exponential_from_fit_and_speech_exponential_quirks_and_clamping():
  # Reference values from tests/misc_math_test.cc (ExponentialFromFitTest).
  got_05 = mappers.exponential_from_fit(0.5, 1.15, 4.68, 0.76)
  got_10 = mappers.exponential_from_fit(1.0, 1.15, 4.68, 0.76)
  assert got_05.dtype == jnp.float32 and got_10.dtype == jnp.float32
  assert float(got_05) == pytest.approx(1.4461764166502666, abs=1e-6)
  assert float(got_10) == pytest.approx(4.2246774455486502, abs=1e-6)

  vec = mappers.exponential_from_fit(
      jnp.array([0.5, 1.0], dtype=jnp.float32), 1.15, 4.68, 0.76
  )
  assert vec.dtype == jnp.float32 and vec.shape == (2,)
  with pytest.raises(ValueError, match="not bool"):
    mappers.exponential_from_fit(0.5, True, 4.68, 0.76)
  with pytest.raises(ValueError, match="real-valued"):
    mappers.exponential_from_fit(0.5, 1.0 + 1.0j, 4.68, 0.76)
  with pytest.raises(ValueError, match="scalar"):
    mappers.exponential_from_fit(0.5, [1.15, 2.0], 4.68, 0.76)

  ones = jnp.ones((21,), dtype=jnp.float32)
  scaled = mappers.predict_speech_exponential(ones, unscaled=False)
  unscaled = mappers.predict_speech_exponential(ones, unscaled=True)
  assert scaled.dtype == jnp.float32 and unscaled.dtype == jnp.float32
  assert float(unscaled) == pytest.approx(4.015861169223797, abs=2e-6)
  assert float(scaled) == pytest.approx(4.999999943193998, abs=2e-6)
  assert float(scaled) <= 5.0

  # Clamping to [1, 5] and jit / vmap behavior.
  low_pred = mappers.predict_speech_exponential(jnp.full((21,), -10.0))
  high_pred = mappers.predict_speech_exponential(jnp.full((21,), 2.0))
  assert float(low_pred) == 1.0
  assert float(high_pred) == 5.0
  batched = jax.jit(jax.vmap(mappers.predict_speech_exponential))(
      jnp.stack([ones, jnp.full((21,), -10.0, dtype=jnp.float32)])
  )
  assert float(batched[0]) == pytest.approx(4.999999943193998, abs=2e-6)
  assert float(batched[1]) == 1.0

  with pytest.raises(ValueError, match="non-empty 1D"):
    mappers.predict_speech_exponential([])
  with pytest.raises(ValueError, match="bool"):
    mappers.predict_speech_exponential(
        ones, unscaled=2  # type: ignore[arg-type]
    )

  # Under x64=True, f64 inputs coerce to f32 and f32 HLO has zero f64 ops.
  assert not jax.config.read("jax_enable_x64")
  with jax.enable_x64(True):
    x21_64 = jnp.ones((21,), dtype=jnp.float64)
    assert (
        mappers.exponential_from_fit(jnp.float64(0.5), 1.15, 4.68, 0.76).dtype
        == jnp.float32
    )
    assert mappers.predict_speech_exponential(x21_64).dtype == jnp.float32
    for fn, args in (
        (
            mappers.exponential_from_fit,
            (
                jnp.float32(0.5),
                jnp.float32(1.15),
                jnp.float32(4.68),
                jnp.float32(0.76),
            ),
        ),
        (mappers.predict_speech_exponential, (ones,)),
    ):
      hlo32 = jax.jit(fn).lower(*args).as_text()
      assert "f64" not in hlo32


def test_check_nonempty_path():
  class _BytesPathLike:

    def __fspath__(self) -> bytes:
      return b"/tmp/model.txt"

  assert _check.check_nonempty_path("model.txt", "p") == "model.txt"
  assert _check.check_nonempty_path(pathlib.Path("a/b"), "p") == "a/b"
  for bad in ("", "a\x00b", b"a", bytearray(b"a"), _BytesPathLike(), 123, None):
    with pytest.raises(ValueError, match="non-empty str or PathLike"):
      _check.check_nonempty_path(bad, "p")  # type: ignore[arg-type]


def test_resolve_model_path_and_read_model_bytes(tmp_path: pathlib.Path):
  sample = tmp_path / "model.txt"
  sample.write_bytes(b"abcd")
  resolved_str, suffix, _, _, size = mappers.resolve_model_path(sample, "p")
  assert suffix == ".txt" and size == 4
  assert mappers.read_model_bytes(resolved_str, max_bytes=4) == b"abcd"
  with pytest.raises(ValueError, match="byte limit"):
    mappers.read_model_bytes(resolved_str, max_bytes=3)

  fifo = tmp_path / "model.fifo"
  os.mkfifo(fifo)
  for non_regular in (tmp_path, pathlib.Path("/dev/null"), fifo):
    with pytest.raises(ValueError, match="regular file"):
      mappers.resolve_model_path(non_regular, "p")
