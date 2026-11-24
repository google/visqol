"""Shared input validation and float32 array coercion helpers."""

import operator
import os
from typing import SupportsIndex

import jax
import jax.numpy as jnp


def is_bool(value: object) -> bool:
  return isinstance(value, bool) or type(value).__name__ in ("bool", "bool_")


def asarray_f32(x: jax.typing.ArrayLike, name: str) -> jax.Array:
  """Validate real-valued numeric input `x` and convert to `float32`."""
  if isinstance(x, jax.Array) and x.dtype == jnp.float32:
    return x
  if is_bool(x):
    raise ValueError(f"{name} must be numeric, not bool")
  if isinstance(x, complex):
    raise ValueError(f"{name} must be real-valued, got {x!r}")
  if isinstance(x, (int, float)):
    try:
      return jnp.asarray(x, dtype=jnp.float32)
    except OverflowError as exc:
      raise ValueError(f"{name} must be numeric, got {x!r}") from exc
  dtype = getattr(x, "dtype", None)
  if dtype is None:
    try:
      x = jnp.asarray(x)
    except (TypeError, ValueError, OverflowError) as exc:
      raise ValueError(f"{name} must be numeric, got {x!r}") from exc
    dtype = x.dtype
  try:
    dt = jnp.dtype(dtype)
  except TypeError as exc:
    raise ValueError(f"{name} must be numeric, got {x!r}") from exc
  if dt == jnp.bool_:
    raise ValueError(f"{name} must be numeric, not bool")
  if jnp.iscomplexobj(x):
    raise ValueError(f"{name} must be real-valued, got dtype {dt}")
  if not jnp.issubdtype(dt, jnp.number):
    raise ValueError(f"{name} must be numeric, got dtype {dt}")
  try:
    return jnp.asarray(x, dtype=jnp.float32)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(f"{name} must be numeric, got {x!r}") from exc


def check_1d_nonempty(x: jax.Array, name: str) -> None:
  if x.ndim != 1 or x.shape[0] == 0:
    raise ValueError(
        f"{name} must be a non-empty 1D array, got shape {x.shape}"
    )


def check_2d_nonempty(x: jax.Array, name: str) -> None:
  if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
    raise ValueError(
        f"{name} must be a non-empty 2D array, got shape {x.shape}"
    )


def _as_int(
    value: SupportsIndex, name: str, *, min_value: int, kind: str
) -> int:
  if is_bool(value) or getattr(value, "dtype", None) == jnp.bool_:
    raise ValueError(f"{name} must be a {kind} int, got {value!r}")
  try:
    iv = operator.index(value)
  except TypeError as exc:
    raise ValueError(f"{name} must be a {kind} int, got {value!r}") from exc
  if iv < min_value:
    raise ValueError(f"{name} must be a {kind} int, got {value!r}")
  return iv


def check_positive_int(value: SupportsIndex, name: str) -> int:
  return _as_int(value, name, min_value=1, kind="positive")


def check_nonneg_int(value: SupportsIndex, name: str) -> int:
  return _as_int(value, name, min_value=0, kind="non-negative")


def check_nonempty_path(path: str | os.PathLike[str], name: str) -> str:
  try:
    raw_path = os.fspath(path)
  except TypeError as exc:
    raise ValueError(f"{name} must be a non-empty str or PathLike") from exc
  if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
    raise ValueError(f"{name} must be a non-empty str or PathLike")
  return raw_path
