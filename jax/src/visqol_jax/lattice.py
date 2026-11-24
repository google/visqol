"""Speech-mode calibrated lattice ensemble similarity-to-quality mapper.

Ports ViSQOL v3.3.3 `tflite_quality_mapper.cc` in pure JAX (`float32`),
loading lattice weights from `.npz` archives produced by
`scripts/export_lattice_npz.py`.
"""

import ast
from collections.abc import Sequence
import dataclasses
import functools
import io
import math
import os
import struct
import zipfile
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from visqol_jax import _check
from visqol_jax import mappers

DEFAULT_LATTICE_TAU = 0.5
_MAX_NPZ_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 10_000

_LATTICE_REQUIRED_KEYS = (
    "in_keypoints",
    "in_inv_lengths",
    "in_has_missing",
    "in_missing_value",
    "lattice_feature_indices",
    "lattice_calibrator_weights",
    "lattice_calibrator_missing_output",
    "lattice_weights",
    "ensemble_weights",
    "out_keypoints",
    "out_inv_lengths",
    "out_weights",
)

LatticeArrays = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
_Int2D = jax.typing.ArrayLike | Sequence[Sequence[int]]
_Bool1D = jax.typing.ArrayLike | Sequence[bool]


def _pack_int32_bounded(
    val: _Int2D, label: str, *, upper: int
) -> tuple[bytes, tuple[int, ...]]:
  """Validate ints in `[0, upper)` and pack to `int32` bytes."""
  arr = mappers.to_host_ndarray(val, label)
  if isinstance(val, (list, tuple)) and mappers.seq_contains_bool(val):
    raise ValueError(f"{label} must have integer dtype, got bool")
  if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
    raise ValueError(f"{label} must have integer dtype, got {arr.dtype}")
  shape = tuple(int(d) for d in arr.shape)
  if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
    raise ValueError(f"{label} must be non-empty 2D, got shape {shape}")
  min_idx, max_idx = int(arr.min()), int(arr.max())
  if min_idx < 0 or max_idx >= upper:
    raise ValueError(
        f"lattice_feature_indices values must be in [0, {upper}), got"
        f" [{min_idx}, {max_idx}]"
    )
  i32 = np.ascontiguousarray(arr, dtype=np.int32)
  return i32.tobytes(), shape


def _pack_bool(val: _Bool1D, label: str) -> tuple[bytes, tuple[int, ...]]:
  """Validate boolean tensor values and pack to native `bytes`."""
  arr = mappers.to_host_ndarray(val, label)
  if arr.dtype != np.bool_:
    raise ValueError(f"{label} must have bool dtype, got {arr.dtype}")
  b = np.ascontiguousarray(arr, dtype=np.bool_)
  return b.tobytes(), tuple(int(d) for d in b.shape)


@dataclasses.dataclass(frozen=True, init=False)
class LatticeModel:
  """Immutable ViSQOL v3.3.3 speech lattice weights backed by Python `bytes`."""

  _buffers: tuple[bytes, ...]
  _shapes: tuple[tuple[int, ...], ...]

  def __init__(
      self,
      in_keypoints: mappers.Float2D,
      in_inv_lengths: mappers.Float2D,
      in_has_missing: _Bool1D,
      in_missing_value: mappers.Float1D,
      lattice_feature_indices: _Int2D,
      lattice_calibrator_weights: mappers.Float3D,
      lattice_calibrator_missing_output: mappers.Float2D,
      lattice_weights: mappers.Float2D,
      ensemble_weights: mappers.Float1D,
      out_keypoints: mappers.Float1D,
      out_inv_lengths: mappers.Float1D,
      out_weights: mappers.Float1D,
  ) -> None:
    buffers: list[bytes] = []
    shapes: list[tuple[int, ...]] = []

    def _add(
        buf_shape: tuple[bytes, tuple[int, ...]],
        key: str,
        expected: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
      buf, shp = buf_shape
      if expected is not None and shp != expected:
        raise ValueError(
            f"Lattice key {key!r} must have shape {expected}, got {shp}"
        )
      buffers.append(buf)
      shapes.append(shp)
      return shp

    in_kp_s = _add(
        mappers.pack_float32(in_keypoints, "Lattice key 'in_keypoints'"),
        "in_keypoints",
    )
    if len(in_kp_s) != 2 or in_kp_s[0] < 5 or in_kp_s[1] < 1:
      raise ValueError(
          "Lattice key 'in_keypoints' must be non-empty 2D with >=5 features,"
          f" got shape {in_kp_s}"
      )
    nf, in_k = in_kp_s
    if (nf - 1) % 4 != 0:
      raise ValueError(
          f"Lattice model feature count must be 4 * num_bands + 1, got {nf}"
      )
    _add(
        mappers.pack_float32(in_inv_lengths, "Lattice key 'in_inv_lengths'"),
        "in_inv_lengths",
        (nf, in_k),
    )
    _add(
        _pack_bool(in_has_missing, "Lattice key 'in_has_missing'"),
        "in_has_missing",
        (nf,),
    )
    _add(
        mappers.pack_float32(
            in_missing_value, "Lattice key 'in_missing_value'"
        ),
        "in_missing_value",
        (nf,),
    )
    feat_s = _add(
        _pack_int32_bounded(
            lattice_feature_indices,
            "Lattice key 'lattice_feature_indices'",
            upper=nf,
        ),
        "lattice_feature_indices",
    )
    nl, rank = feat_s

    _add(
        mappers.pack_float32(
            lattice_calibrator_weights,
            "Lattice key 'lattice_calibrator_weights'",
        ),
        "lattice_calibrator_weights",
        (nl, rank, in_k + 1),
    )
    _add(
        mappers.pack_float32(
            lattice_calibrator_missing_output,
            "Lattice key 'lattice_calibrator_missing_output'",
        ),
        "lattice_calibrator_missing_output",
        (nl, rank),
    )
    lat_w_s = _add(
        mappers.pack_float32(lattice_weights, "Lattice key 'lattice_weights'"),
        "lattice_weights",
    )
    if (
        len(lat_w_s) != 2
        or lat_w_s[0] != nl
        or rank > 30
        or lat_w_s[1] != (1 << rank)
    ):
      raise ValueError(
          f"Lattice key 'lattice_weights' must have shape ({nl}, 2**{rank}),"
          f" got {lat_w_s}"
      )
    _add(
        mappers.pack_float32(
            ensemble_weights, "Lattice key 'ensemble_weights'"
        ),
        "ensemble_weights",
        (nl,),
    )

    out_kp_s = _add(
        mappers.pack_float32(out_keypoints, "Lattice key 'out_keypoints'"),
        "out_keypoints",
    )
    if len(out_kp_s) != 1 or out_kp_s[0] < 1:
      raise ValueError(
          "Lattice key 'out_keypoints' must be non-empty 1D, got shape"
          f" {out_kp_s}"
      )
    out_k = out_kp_s[0]
    _add(
        mappers.pack_float32(out_inv_lengths, "Lattice key 'out_inv_lengths'"),
        "out_inv_lengths",
        (out_k,),
    )
    _add(
        mappers.pack_float32(out_weights, "Lattice key 'out_weights'"),
        "out_weights",
        (out_k + 1,),
    )

    object.__setattr__(self, "_buffers", tuple(buffers))
    object.__setattr__(self, "_shapes", tuple(shapes))

  def _arr(self, idx: int, dtype: jax.typing.DTypeLike) -> jax.Array:
    return jnp.frombuffer(self._buffers[idx], dtype=dtype).reshape(
        self._shapes[idx]
    )

  @property
  def num_bands(self) -> int:
    return (self._shapes[0][0] - 1) // 4

  @property
  def in_keypoints(self) -> jax.Array:
    return self._arr(0, jnp.float32)

  @property
  def in_inv_lengths(self) -> jax.Array:
    return self._arr(1, jnp.float32)

  @property
  def in_has_missing(self) -> jax.Array:
    return self._arr(2, jnp.bool_)

  @property
  def in_missing_value(self) -> jax.Array:
    return self._arr(3, jnp.float32)

  @property
  def lattice_feature_indices(self) -> jax.Array:
    return self._arr(4, jnp.int32)

  @property
  def lattice_calibrator_weights(self) -> jax.Array:
    return self._arr(5, jnp.float32)

  @property
  def lattice_calibrator_missing_output(self) -> jax.Array:
    return self._arr(6, jnp.float32)

  @property
  def lattice_weights(self) -> jax.Array:
    return self._arr(7, jnp.float32)

  @property
  def ensemble_weights(self) -> jax.Array:
    return self._arr(8, jnp.float32)

  @property
  def out_keypoints(self) -> jax.Array:
    return self._arr(9, jnp.float32)

  @property
  def out_inv_lengths(self) -> jax.Array:
    return self._arr(10, jnp.float32)

  @property
  def out_weights(self) -> jax.Array:
    return self._arr(11, jnp.float32)


def _read_npy_header(
    buf: io.BytesIO,
) -> tuple[tuple[int, ...], np.dtype[np.generic]]:
  """Read a `.npy` v1.0/v2.0/v3.0 header without allocating array data."""
  version = np.lib.format.read_magic(buf)
  if version == (1, 0):
    shape, _, dtype = np.lib.format.read_array_header_1_0(
        buf, max_header_size=_MAX_NPY_HEADER_BYTES
    )
    return shape, dtype
  if version == (2, 0):
    shape, _, dtype = np.lib.format.read_array_header_2_0(
        buf, max_header_size=_MAX_NPY_HEADER_BYTES
    )
    return shape, dtype
  if version == (3, 0):
    len_bytes = buf.read(4)
    if len(len_bytes) != 4:
      raise ValueError("Truncated .npy v3.0 header length")
    (header_len,) = struct.unpack("<I", len_bytes)
    if header_len > _MAX_NPY_HEADER_BYTES:
      raise ValueError("Oversized .npy v3.0 header")
    raw_header = buf.read(header_len)
    if len(raw_header) != header_len:
      raise ValueError("Truncated .npy v3.0 header")
    header = ast.literal_eval(raw_header.decode("utf-8"))
    if (
        not isinstance(header, dict)
        or header.keys() != np.lib.format.EXPECTED_KEYS
        or not isinstance(header["shape"], tuple)
        or not isinstance(header["fortran_order"], bool)
    ):
      raise ValueError("Invalid .npy v3.0 header")
    return header["shape"], np.lib.format.descr_to_dtype(header["descr"])
  raise ValueError(f"Unsupported .npy version {version!r}")


def _read_npy_member(
    member_bytes: bytes, *, source: str, key: str
) -> np.ndarray:
  """Validate a `.npy` header against `member_bytes` before allocating."""
  buf = io.BytesIO(member_bytes)
  try:
    shape, dtype = _read_npy_header(buf)
    if (
        dtype.hasobject
        or dtype.itemsize <= 0
        or any(
            isinstance(d, bool) or not isinstance(d, int) or d < 0
            for d in shape
        )
        or math.prod(shape) * dtype.itemsize > len(member_bytes) - buf.tell()
    ):
      raise ValueError("Invalid .npy header")
    buf.seek(0)
    return np.lib.format.read_array(
        buf, allow_pickle=False, max_header_size=_MAX_NPY_HEADER_BYTES
    )
  except (
      ValueError,
      SyntaxError,
      TypeError,
      OverflowError,
      RecursionError,
      EOFError,
      OSError,
  ) as exc:
    raise ValueError(
        f"Corrupt .npz lattice model {source!r} at key {key!r}"
    ) from exc


def _lattice_from_npz_bytes(
    raw_bytes: bytes, *, source: str, num_bands: int | None = None
) -> LatticeModel:
  """Construct and validate a `LatticeModel` from `.npz` archive bytes."""
  if not raw_bytes:
    raise ValueError(f"Corrupt .npz lattice model: {source!r}")
  if raw_bytes.startswith(np.lib.format.MAGIC_PREFIX):
    raise ValueError(f"Lattice model {source!r} must be a .npz archive")
  try:
    zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
  except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
    raise ValueError(f"Corrupt .npz lattice model: {source!r}") from exc
  with zf:
    infos = zf.infolist()
    key_to_info: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
      key = info.filename.removesuffix(".npy")
      if key in key_to_info:
        raise ValueError(
            f"Corrupt .npz lattice model {source!r} at key {key!r}"
        )
      key_to_info[key] = info
    missing = [k for k in _LATTICE_REQUIRED_KEYS if k not in key_to_info]
    if missing:
      raise ValueError(
          f"Lattice model missing required keys: {', '.join(missing)}"
      )
    unknown = sorted(set(key_to_info) - set(_LATTICE_REQUIRED_KEYS))
    if unknown:
      raise ValueError(
          f"Lattice model has unexpected keys: {', '.join(unknown)}"
      )
    total_size = 0
    for info in infos:
      total_size += info.file_size
      if info.file_size < 0 or total_size > _MAX_NPZ_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"Lattice model {source!r} uncompressed size exceeds"
            f" {_MAX_NPZ_UNCOMPRESSED_BYTES} byte limit"
        )
    remaining = _MAX_NPZ_UNCOMPRESSED_BYTES
    data: dict[str, np.ndarray] = {}
    for key in _LATTICE_REQUIRED_KEYS:
      try:
        with zf.open(key_to_info[key]) as member:
          member_bytes = member.read(remaining + 1)
      except (
          zipfile.BadZipFile,
          OSError,
          EOFError,
          ValueError,
          zlib.error,
      ) as exc:
        raise ValueError(
            f"Corrupt .npz lattice model {source!r} at key {key!r}"
        ) from exc
      if len(member_bytes) > remaining:
        raise ValueError(
            f"Lattice model {source!r} uncompressed size exceeds"
            f" {_MAX_NPZ_UNCOMPRESSED_BYTES} byte limit"
        )
      remaining -= len(member_bytes)
      data[key] = _read_npy_member(member_bytes, source=source, key=key)

  lat = LatticeModel(
      in_keypoints=data["in_keypoints"],
      in_inv_lengths=data["in_inv_lengths"],
      in_has_missing=data["in_has_missing"],
      in_missing_value=data["in_missing_value"],
      lattice_feature_indices=data["lattice_feature_indices"],
      lattice_calibrator_weights=data["lattice_calibrator_weights"],
      lattice_calibrator_missing_output=data[
          "lattice_calibrator_missing_output"
      ],
      lattice_weights=data["lattice_weights"],
      ensemble_weights=data["ensemble_weights"],
      out_keypoints=data["out_keypoints"],
      out_inv_lengths=data["out_inv_lengths"],
      out_weights=data["out_weights"],
  )
  if num_bands is not None and lat.num_bands != num_bands:
    raise ValueError(
        f"Lattice model has {lat.num_bands} bands, expected {num_bands} bands"
    )
  return lat


def _check_lattice_bands(lat: LatticeModel, num_bands: int | None) -> None:
  if num_bands is not None and lat.num_bands != num_bands:
    raise ValueError(
        f"Bundled lattice model has {lat.num_bands} bands, expected"
        f" {num_bands} bands"
    )


# Cache pure Python bytes/ints only so JAX tracers never leak across jit.
@functools.cache
def _load_default_lattice_model() -> LatticeModel:
  """Load and cache the bundled ViSQOL v3.3.3 speech lattice model."""
  return _lattice_from_npz_bytes(
      mappers.MODELS.joinpath("speech_lattice.npz").read_bytes(),
      source="speech_lattice.npz",
  )


@functools.lru_cache(maxsize=16)
def _load_custom_lattice_model(
    path_str: str,
    suffix: str,
    ino: int,
    mtime_ns: int,
    size: int,
    num_bands: int | None,
) -> LatticeModel:
  """Load and cache a custom `.npz` lattice model."""
  del ino, mtime_ns, size
  if suffix == ".npz":
    return _lattice_from_npz_bytes(
        mappers.read_model_bytes(
            path_str, max_bytes=_MAX_NPZ_UNCOMPRESSED_BYTES
        ),
        source=path_str,
        num_bands=num_bands,
    )
  if suffix == ".tflite":
    raise ValueError(
        f"Unsupported lattice model format '.tflite' for {path_str!r}; convert"
        " TF-Lattice .tflite models to .npz using scripts/export_lattice_npz.py"
        " or pass path=None for the bundled speech model."
    )
  raise ValueError(
      f"Unsupported lattice model format {suffix!r} for {path_str!r}; expected"
      " a .npz lattice model."
  )


def load_lattice_model(
    path: str | os.PathLike[str] | None = None,
    *,
    num_bands: int | None = None,
) -> LatticeModel:
  """Load bundled speech lattice weights or a custom `.npz` archive."""
  nb = (
      None
      if num_bands is None
      else _check.check_positive_int(num_bands, "num_bands")
  )
  if path is None:
    lat = _load_default_lattice_model()
    _check_lattice_bands(lat, nb)
    return lat
  resolved_str, suffix, ino, mtime_ns, size = mappers.resolve_model_path(
      path, "path"
  )
  return _load_custom_lattice_model(
      resolved_str, suffix, ino, mtime_ns, size, nb
  )


def lattice_arrays(lat: LatticeModel) -> LatticeArrays:
  """Unpack `LatticeModel` into a flat tuple of `jax.Array`s for JIT."""
  return (
      lat.in_keypoints,
      lat.in_inv_lengths,
      lat.in_has_missing,
      lat.in_missing_value,
      lat.lattice_feature_indices,
      lat.lattice_calibrator_weights,
      lat.lattice_calibrator_missing_output,
      lat.lattice_weights,
      lat.ensemble_weights,
      lat.out_keypoints,
      lat.out_inv_lengths,
      lat.out_weights,
  )


def eval_lattice(
    fvnsim: jax.Array,
    fvnsim10: jax.Array,
    fstdnsim: jax.Array,
    fvdegenergy: jax.Array,
    tau: jax.Array,
    in_kp: jax.Array,
    in_inv_len: jax.Array,
    in_has_missing: jax.Array,
    in_missing_value: jax.Array,
    feat_idx: jax.Array,
    cal_w: jax.Array,
    cal_miss_out: jax.Array,
    lat_w: jax.Array,
    ens_w: jax.Array,
    out_kp: jax.Array,
    out_inv_len: jax.Array,
    out_w: jax.Array,
) -> jax.Array:
  """Evaluate CalibratedLatticeEnsemble from traced JAX arrays in `float32`."""
  x32 = jnp.concatenate(
      [
          fvnsim.astype(jnp.float32),
          fvnsim10.astype(jnp.float32),
          fstdnsim.astype(jnp.float32),
          fvdegenergy.astype(jnp.float32),
          tau.astype(jnp.float32)[None],
      ],
      axis=0,
  )
  one = jnp.float32(1.0)
  zero = jnp.float32(0.0)

  slopes = jnp.clip((x32[:, None] - in_kp) * in_inv_len, zero, one)
  basis = jnp.concatenate(
      [jnp.ones((x32.shape[0], 1), dtype=jnp.float32), slopes], axis=1
  )

  c = jnp.sum(basis[feat_idx] * cal_w, axis=-1)
  is_miss = in_has_missing & (x32 == in_missing_value)
  c = jnp.where(is_miss[feat_idx], cal_miss_out, c)

  w0 = one - jnp.minimum(jnp.abs(c), one)
  w1 = one - jnp.minimum(jnp.abs(c - one), one)
  cur = lat_w
  for d in range(c.shape[1]):
    cur = cur.reshape(cur.shape[0], 2, -1)
    cur = cur[:, 0, :] * w0[:, d, None] + cur[:, 1, :] * w1[:, d, None]
  lat_out = cur[:, 0]

  z = jnp.sum(lat_out * ens_w, axis=-1)
  out_slopes = jnp.clip((z - out_kp) * out_inv_len, zero, one)
  out_basis = jnp.concatenate([jnp.ones((1,), dtype=jnp.float32), out_slopes])
  return jnp.sum(out_basis * out_w, axis=-1)


def predict_lattice(
    fvnsim: jax.typing.ArrayLike,
    fvnsim10: jax.typing.ArrayLike,
    fstdnsim: jax.typing.ArrayLike,
    fvdegenergy: jax.typing.ArrayLike,
    tau: float | jax.typing.ArrayLike = DEFAULT_LATTICE_TAU,
    model: LatticeModel | None = None,
    *,
    svr_model_path: str | os.PathLike[str] | None = None,
) -> jax.Array:
  """Predict speech-mode MOS-LQO via the ViSQOL v3.3.3 lattice ensemble."""
  if model is not None and svr_model_path is not None:
    raise ValueError("Pass at most one of model or svr_model_path")
  if model is not None and not isinstance(model, LatticeModel):
    raise TypeError(
        f"Expected LatticeModel or None, got {type(model).__name__}"
    )
  lat = load_lattice_model(svr_model_path) if model is None else model
  num_bands = lat.num_bands

  parts: list[jax.Array] = []
  for name, arr in (
      ("fvnsim", fvnsim),
      ("fvnsim10", fvnsim10),
      ("fstdnsim", fstdnsim),
      ("fvdegenergy", fvdegenergy),
  ):
    vec = _check.asarray_f32(arr, name)
    if vec.ndim != 1 or vec.shape[0] != num_bands:
      raise ValueError(
          f"{name} must be a 1D array of length {num_bands}, got shape"
          f" {vec.shape}"
      )
    parts.append(vec)
  tau_arr = mappers.as_scalar_f32(tau, "tau", require_finite=True)

  return eval_lattice(
      parts[0],
      parts[1],
      parts[2],
      parts[3],
      tau_arr,
      *lattice_arrays(lat),
  )
