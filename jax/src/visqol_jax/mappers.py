"""Shared quality-mapper helpers and speech exponential MOS mapper.

Ports ViSQOL v3.3.3 `misc_math.cc` (`ExponentialFromFit`) and
`speech_similarity_to_quality_mapper.cc`, and holds model-loading helpers
shared by `svr.py` and `lattice.py`.
"""

from collections.abc import Sequence
import importlib.resources
import math
import os
import pathlib
import stat
import struct

import jax
import jax.numpy as jnp
import numpy as np

from visqol_jax import _check

SPEECH_EXP_A = -262.847869
SPEECH_EXP_B = 0.0154302525
SPEECH_EXP_X0 = -361.063949
SPEECH_EXP_SCALE = 1.245063
MOS_MIN = 1.0
MOS_MAX = 5.0

_SPEECH_EXP_C1 = math.exp(-SPEECH_EXP_B * SPEECH_EXP_X0)
_SPEECH_EXP_C0 = SPEECH_EXP_A + _SPEECH_EXP_C1

# Package-relative lookup works across runfiles, editable, and wheel installs.
MODELS = importlib.resources.files(__package__).joinpath("_model_data")

Float1D = jax.typing.ArrayLike | Sequence[float]
Float2D = jax.typing.ArrayLike | Sequence[Sequence[float]]
Float3D = jax.typing.ArrayLike | Sequence[Sequence[Sequence[float]]]
FloatTensor = Float1D | Float2D | Float3D

__all__ = [
    "MOS_MAX",
    "MOS_MIN",
    "SPEECH_EXP_A",
    "SPEECH_EXP_B",
    "SPEECH_EXP_SCALE",
    "SPEECH_EXP_X0",
    "exponential_from_fit",
    "predict_speech_exponential",
]


def resolve_model_path(
    path: str | os.PathLike[str], name: str
) -> tuple[str, str, int, int, int]:
  """Resolve `path` while keeping its logical suffix across CAS symlinks."""
  raw_path = _check.check_nonempty_path(path, name)
  resolved = pathlib.Path(raw_path).resolve()
  suffix = pathlib.Path(raw_path).suffix.lower() or resolved.suffix.lower()
  st = resolved.stat()
  if not stat.S_ISREG(st.st_mode):
    raise ValueError(f"{name} must be a regular file: {raw_path!r}")
  return str(resolved), suffix, st.st_ino, st.st_mtime_ns, st.st_size


def read_model_bytes(path_str: str, *, max_bytes: int) -> bytes:
  """Read at most `max_bytes` from `path_str` or raise `ValueError`."""
  with open(path_str, "rb") as fh:
    data = fh.read(max_bytes + 1)
  if len(data) > max_bytes:
    raise ValueError(f"Model file {path_str!r} exceeds {max_bytes} byte limit")
  return data


def finite_f32_float(val: float, label: str) -> float:
  """Validate finite real-valued scalar `val` within `float32` range."""
  if _check.is_bool(val) or not isinstance(val, (int, float)):
    raise ValueError(f"{label} must be a numeric scalar, got {val!r}")
  try:
    fv = float(val)
    if not math.isfinite(fv):
      raise ValueError(f"{label} must be finite, got {fv}")
    struct.pack("=f", fv)
  except (OverflowError, struct.error) as exc:
    raise ValueError(f"{label} must be finite in float32, got {val!r}") from exc
  return fv


def as_scalar_f32(
    value: float | jax.typing.ArrayLike,
    name: str,
    *,
    require_finite: bool = False,
) -> jax.Array:
  """Validate scalar numeric parameter and convert to `float32`."""
  arr = _check.asarray_f32(value, name)
  if arr.ndim != 0:
    raise ValueError(f"{name} must be a numeric scalar, got shape {arr.shape}")
  if (
      require_finite
      and not isinstance(arr, jax.core.Tracer)
      and not math.isfinite(float(arr))
  ):
    raise ValueError(f"{name} must be a finite float32 scalar, got {value!r}")
  return arr


def seq_contains_bool(obj: object) -> bool:
  if isinstance(obj, (list, tuple)):
    return any(seq_contains_bool(x) for x in obj)
  return _check.is_bool(obj)


def to_host_ndarray(val: object, label: str) -> np.ndarray:
  """Convert a concrete array or rectangular sequence to a NumPy array."""
  if isinstance(val, jax.core.Tracer):
    raise ValueError(f"{label} must not be a JAX Tracer")
  if isinstance(val, (bytes, bytearray, memoryview, str, dict, set)):
    raise ValueError(
        f"{label} must be a sequence or array, got {type(val).__name__}"
    )
  try:
    return np.asarray(val)
  except ValueError as exc:
    raise ValueError(
        f"{label} must be a rectangular sequence, not ragged"
    ) from exc


def pack_float32(
    val: FloatTensor, label: str, *, allow_int_array: bool = False
) -> tuple[bytes, tuple[int, ...]]:
  """Validate finite `float32` tensor values and pack to native `bytes`."""
  arr = to_host_ndarray(val, label)
  if isinstance(val, (list, tuple)) and seq_contains_bool(val):
    raise ValueError(f"{label} must have floating dtype, got bool")
  dt = arr.dtype
  if dt == np.bool_ or np.issubdtype(dt, np.complexfloating):
    raise ValueError(f"{label} must have floating dtype, got {dt}")
  if not np.issubdtype(dt, np.floating):
    is_python_seq = isinstance(val, (list, tuple))
    if not (
        (allow_int_array or is_python_seq) and np.issubdtype(dt, np.integer)
    ):
      if dt == np.object_:
        if all(
            isinstance(x, (int, float)) and not _check.is_bool(x)
            for x in arr.flat
        ):
          raise ValueError(f"{label} must contain finite float32 values")
      raise ValueError(f"{label} must have floating dtype, got {dt}")
  if arr.size > 0 and not bool(np.all(np.isfinite(arr))):
    raise ValueError(f"{label} must contain finite values")
  with np.errstate(over="ignore"):
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
  if f32.size > 0 and not bool(np.all(np.isfinite(f32))):
    raise ValueError(f"{label} must contain finite float32 values")
  return f32.tobytes(), tuple(int(d) for d in f32.shape)


def exponential_from_fit(
    x: jax.typing.ArrayLike,
    a: float | jax.typing.ArrayLike,
    b: float | jax.typing.ArrayLike,
    x0: float | jax.typing.ArrayLike,
) -> jax.Array:
  """Evaluate `a + exp(b * (x - x0))` (`MiscMath::ExponentialFromFit`)."""
  av = as_scalar_f32(a, "a")
  bv = as_scalar_f32(b, "b")
  x0v = as_scalar_f32(x0, "x0")
  xv = _check.asarray_f32(x, "x")
  return av + jnp.exp(bv * (xv - x0v))


def predict_speech_exponential(
    fvnsim: jax.typing.ArrayLike, unscaled: bool = False
) -> jax.Array:
  """Predict speech-mode MOS-LQO via the TCD-VOIP exponential fit (`[1, 5]`)."""
  if not _check.is_bool(unscaled):
    raise ValueError(f"unscaled must be a bool, got {unscaled!r}")
  x = _check.asarray_f32(fvnsim, "fvnsim")
  _check.check_1d_nonempty(x, "fvnsim")
  # Evaluate (a + exp(-b * x0)) + exp(-b * x0) * expm1(b * x) to avoid f32 loss.
  mos = jnp.float32(_SPEECH_EXP_C0) + jnp.float32(_SPEECH_EXP_C1) * jnp.expm1(
      jnp.float32(SPEECH_EXP_B) * jnp.mean(x)
  )
  scale = jnp.float32(1.0 if bool(unscaled) else SPEECH_EXP_SCALE)
  return jnp.clip(mos * scale, jnp.float32(MOS_MIN), jnp.float32(MOS_MAX))
