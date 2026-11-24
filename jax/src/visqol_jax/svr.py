"""Audio-mode RBF nu-SVR similarity-to-quality mapper.

Ports ViSQOL v3.3.3 `svr_similarity_to_quality_mapper.cc` and
`support_vector_regression_model.cc`.
"""

import dataclasses
import functools
import gzip
import io
import math
import os
import struct
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from visqol_jax import _check
from visqol_jax import mappers

_MAX_GZIP_BYTES = 64 * 1024 * 1024
_MAX_SVR_FEATURES = 65_536
_MAX_SVR_ELEMENTS = 16_777_216


def _decompress_gzip(
    raw_bytes: bytes, source: str, *, max_bytes: int = _MAX_GZIP_BYTES
) -> bytes:
  """Decompress a bounded non-empty gzip stream or raise `ValueError`."""
  if not raw_bytes:
    raise ValueError(f"Corrupt gzip SVR model: {source!r}")
  try:
    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
      data = gz.read(max_bytes + 1)
  except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as exc:
    raise ValueError(f"Corrupt gzip SVR model: {source!r}") from exc
  if len(data) > max_bytes:
    raise ValueError(
        f"Decompressed SVR model {source!r} exceeds {max_bytes} byte limit"
    )
  return data


@dataclasses.dataclass(frozen=True, init=False)
class SvrModel:
  """Immutable RBF SVR model backed by Python `float`s and `float32` `bytes`."""

  gamma: float
  rho: float
  _sv_coef_bytes: bytes
  _support_vectors_bytes: bytes
  _sv_shape: tuple[int, int]

  def __init__(
      self,
      gamma: float,
      rho: float,
      sv_coef: mappers.Float1D,
      support_vectors: mappers.Float2D,
  ) -> None:
    checked_gamma = mappers.finite_f32_float(gamma, "SVR key 'gamma'")
    if checked_gamma < 0.0:
      raise ValueError(
          f"SVR key 'gamma' must be non-negative, got {checked_gamma}"
      )
    checked_rho = mappers.finite_f32_float(rho, "SVR key 'rho'")
    coef_b, coef_shape = mappers.pack_float32(
        sv_coef, "SVR key 'sv_coef'", allow_int_array=True
    )
    sv_b, sv_shape = mappers.pack_float32(
        support_vectors, "SVR key 'support_vectors'", allow_int_array=True
    )
    if len(sv_shape) != 2 or sv_shape[0] == 0 or sv_shape[1] == 0:
      raise ValueError(
          "SVR key 'support_vectors' must be a non-empty 2D array, got shape"
          f" {sv_shape}"
      )
    if coef_shape != (sv_shape[0],):
      raise ValueError(
          f"SVR key 'sv_coef' must have shape ({sv_shape[0]},), got"
          f" {coef_shape}"
      )
    object.__setattr__(self, "gamma", checked_gamma)
    object.__setattr__(self, "rho", checked_rho)
    object.__setattr__(self, "_sv_coef_bytes", coef_b)
    object.__setattr__(self, "_support_vectors_bytes", sv_b)
    object.__setattr__(self, "_sv_shape", (sv_shape[0], sv_shape[1]))

  @property
  def num_sv(self) -> int:
    return self._sv_shape[0]

  @property
  def num_features(self) -> int:
    return self._sv_shape[1]

  @property
  def sv_coef(self) -> jax.Array:
    return jnp.frombuffer(self._sv_coef_bytes, dtype=jnp.float32)

  @property
  def support_vectors(self) -> jax.Array:
    return jnp.frombuffer(
        self._support_vectors_bytes, dtype=jnp.float32
    ).reshape(self._sv_shape)


def _parse_libsvm_svr_text(
    text: str, *, num_features: int | None = None
) -> SvrModel:
  """Parse libsvm RBF SVR text content into an immutable `SvrModel`."""
  header: dict[str, str] = {}
  sv_lines: list[str] = []
  in_sv = False
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if in_sv:
      sv_lines.append(line)
    elif line == "SV":
      in_sv = True
    else:
      parts = line.split(None, 1)
      if len(parts) != 2:
        raise ValueError(f"Malformed libsvm header line: {line!r}")
      header[parts[0]] = parts[1]

  svm_type = header.get("svm_type")
  if svm_type not in ("nu_svr", "epsilon_svr"):
    raise ValueError(
        f"Expected svm_type nu_svr or epsilon_svr, got {svm_type!r}"
    )
  if header.get("kernel_type") != "rbf":
    raise ValueError(
        f"Expected kernel_type rbf, got {header.get('kernel_type')!r}"
    )
  if "nr_class" in header and int(header["nr_class"]) != 2:
    raise ValueError(f"Expected nr_class 2 for SVR, got {header['nr_class']!r}")
  for required in ("gamma", "rho", "total_sv"):
    if required not in header:
      raise ValueError(f"Missing required libsvm header field: {required}")
  if not in_sv:
    raise ValueError("Missing SV section in libsvm SVR model")

  total_sv = int(header["total_sv"])
  if total_sv <= 0:
    raise ValueError(f"Expected positive total_sv, got {total_sv}")
  if len(sv_lines) != total_sv:
    raise ValueError(
        f"Expected {total_sv} support vectors, got {len(sv_lines)}"
    )

  gamma = mappers.finite_f32_float(float(header["gamma"]), "SVR key 'gamma'")
  if gamma < 0.0:
    raise ValueError(f"SVR key 'gamma' must be non-negative, got {gamma}")
  rho = mappers.finite_f32_float(float(header["rho"]), "SVR key 'rho'")

  sv_coef: list[float] = []
  parsed_rows: list[list[tuple[int, float]]] = []
  max_feat_idx = 0
  for line in sv_lines:
    tokens = line.split()
    coef = float(tokens[0])
    if not math.isfinite(coef):
      raise ValueError(f"Non-finite libsvm sv_coef: {coef}")
    coef = mappers.finite_f32_float(coef, "SVR key 'sv_coef'")
    pairs: list[tuple[int, float]] = []
    extra_sq = 0.0
    prev_idx = -1
    for token in tokens[1:]:
      if ":" not in token:
        raise ValueError(f"Malformed libsvm feature token: {token!r}")
      idx_s, val_s = token.split(":", 1)
      idx = int(idx_s)
      if idx < 0:
        raise ValueError(
            f"Expected non-negative libsvm feature index, got {idx}"
        )
      if idx <= prev_idx:
        raise ValueError(
            "Expected strictly increasing libsvm feature indices, got"
            f" {idx} after {prev_idx}"
        )
      prev_idx = idx
      val = float(val_s)
      if not math.isfinite(val):
        raise ValueError(f"Non-finite libsvm feature value: {val}")
      try:
        struct.pack("=f", val)
      except OverflowError as exc:
        raise ValueError(
            f"Libsvm feature value must be finite in float32, got {val}"
        ) from exc
      if idx == 0:
        # Index 0 in legacy cpp_model.txt pairs with x[0] == 0 in libsvm.
        extra_sq += val * val
      else:
        max_feat_idx = max(max_feat_idx, idx)
        pairs.append((idx - 1, val))
    sv_coef.append(
        coef * math.exp(-gamma * extra_sq)
        if gamma > 0.0 and extra_sq > 0.0
        else coef
    )
    parsed_rows.append(pairs)

  if num_features is None:
    if max_feat_idx == 0:
      raise ValueError("Libsvm SVR model contains no feature indices >= 1")
    if max_feat_idx > _MAX_SVR_FEATURES:
      raise ValueError(
          f"Libsvm SVR feature index {max_feat_idx} exceeds limit"
          f" {_MAX_SVR_FEATURES}"
      )
    width = max_feat_idx
  else:
    if max_feat_idx > num_features:
      raise ValueError(
          f"Libsvm SVR feature index {max_feat_idx} exceeds expected"
          f" {num_features}"
      )
    width = num_features

  if total_sv * width > _MAX_SVR_ELEMENTS:
    raise ValueError(
        f"Libsvm SVR matrix size ({total_sv} x {width}) exceeds limit"
        f" {_MAX_SVR_ELEMENTS} elements"
    )

  support_vectors = np.zeros((total_sv, width), dtype=np.float32)
  for row_idx, pairs in enumerate(parsed_rows):
    for col_idx, val in pairs:
      support_vectors[row_idx, col_idx] = np.float32(val)

  return SvrModel(
      gamma, rho, np.asarray(sv_coef, dtype=np.float32), support_vectors
  )


def _parse_svr_bytes(
    raw_bytes: bytes,
    *,
    source: str,
    compressed: bool,
    num_features: int | None,
) -> SvrModel:
  """Decompress if needed, decode UTF-8, and parse libsvm SVR bytes."""
  if compressed:
    raw_bytes = _decompress_gzip(raw_bytes, source)
  try:
    text = raw_bytes.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise ValueError(
        f"Libsvm SVR model {source!r} is not valid UTF-8 text"
    ) from exc
  return _parse_libsvm_svr_text(text, num_features=num_features)


def _parse_svr_file(
    path_str: str, suffix: str, *, num_features: int | None
) -> SvrModel:
  """Read and parse a `.txt` or `.txt.gz` libsvm SVR file."""
  raw_bytes = mappers.read_model_bytes(path_str, max_bytes=_MAX_GZIP_BYTES)
  compressed = suffix.endswith(".gz") or (
      not suffix and raw_bytes.startswith(b"\x1f\x8b")
  )
  return _parse_svr_bytes(
      raw_bytes,
      source=path_str,
      compressed=compressed,
      num_features=num_features,
  )


def parse_libsvm_svr(
    path: str | os.PathLike[str], *, num_features: int | None = None
) -> SvrModel:
  """Parse a libsvm RBF SVR `.txt` or `.txt.gz` file into an `SvrModel`."""
  nf = (
      None
      if num_features is None
      else _check.check_positive_int(num_features, "num_features")
  )
  resolved_str, suffix, _, _, _ = mappers.resolve_model_path(path, "path")
  return _parse_svr_file(resolved_str, suffix, num_features=nf)


# Cache pure Python bytes/floats only so JAX tracers never leak across jit.
@functools.cache
def _load_default_svr_model() -> SvrModel:
  """Load and cache the bundled ViSQOL v3.3.3 audio nu-SVR model."""
  return _parse_svr_bytes(
      mappers.MODELS.joinpath("svr_model.txt.gz").read_bytes(),
      source="svr_model.txt.gz",
      compressed=True,
      num_features=None,
  )


@functools.lru_cache(maxsize=16)
def _load_custom_svr_model(
    path_str: str,
    suffix: str,
    ino: int,
    mtime_ns: int,
    size: int,
    num_features: int | None,
) -> SvrModel:
  del ino, mtime_ns, size
  return _parse_svr_file(path_str, suffix, num_features=num_features)


def load_svr_model(
    path: str | os.PathLike[str] | None = None,
    *,
    num_features: int | None = None,
) -> SvrModel:
  """Load the bundled audio nu-SVR model or a custom `.txt` / `.txt.gz` file."""
  nf = (
      None
      if num_features is None
      else _check.check_positive_int(num_features, "num_features")
  )
  if path is None:
    svr = _load_default_svr_model()
    if nf is not None and svr.num_features != nf:
      raise ValueError(
          f"Bundled SVR model has {svr.num_features} features, expected {nf}"
      )
    return svr
  resolved_str, suffix, ino, mtime_ns, size = mappers.resolve_model_path(
      path, "svr_model_path"
  )
  return _load_custom_svr_model(resolved_str, suffix, ino, mtime_ns, size, nf)


def eval_svr(
    fvnsim: jax.Array,
    sv: jax.Array,
    sv_coef: jax.Array,
    gamma: float | jax.Array,
    rho: float | jax.Array,
) -> jax.Array:
  """Evaluate RBF SVR in `float32` and clip to `[MOS_MIN, MOS_MAX]`."""
  diff = fvnsim[None, :] - sv
  sq_dist = jnp.sum(diff * diff, axis=-1)
  gamma_f32 = jnp.asarray(gamma, dtype=jnp.float32)
  rho_f32 = jnp.asarray(rho, dtype=jnp.float32)
  raw = jnp.sum(sv_coef * jnp.exp(-gamma_f32 * sq_dist)) - rho_f32
  return jnp.clip(
      raw, jnp.float32(mappers.MOS_MIN), jnp.float32(mappers.MOS_MAX)
  )


def predict_svr(
    fvnsim: jax.typing.ArrayLike,
    model: SvrModel | None = None,
    *,
    svr_model_path: str | os.PathLike[str] | None = None,
) -> jax.Array:
  """Predict audio-mode MOS-LQO from per-band `fvnsim` via RBF nu-SVR."""
  if model is not None and svr_model_path is not None:
    raise ValueError("Pass at most one of model or svr_model_path")
  if model is not None and not isinstance(model, SvrModel):
    raise TypeError(f"Expected SvrModel or None, got {type(model).__name__}")
  x = _check.asarray_f32(fvnsim, "fvnsim")
  _check.check_1d_nonempty(x, "fvnsim")
  if model is None:
    nf = x.shape[0] if svr_model_path is not None else None
    svr = load_svr_model(svr_model_path, num_features=nf)
  else:
    svr = model
  if x.shape[0] != svr.num_features:
    raise ValueError(
        f"fvnsim must be a 1D array of length {svr.num_features}, got shape"
        f" {x.shape}"
    )
  return eval_svr(x, svr.support_vectors, svr.sv_coef, svr.gamma, svr.rho)
