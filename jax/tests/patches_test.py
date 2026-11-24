"""Tests for patches.py against C++ ViSQOL v3.3.3."""

import math
import re

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import patches
from visqol_jax import spectrogram

# Exact vectors and constants from tests/rms_vad_test.cc.
_CPP_VAD_TOLERANCE = 0.0001
_CPP_CHUNK = (
    186, 236, 44, -152, -155, -2, 66, 5, -108, -107, 14, 141, 151, 31, -90
)
_CPP_CHUNK_RMS = 120.7736
_CPP_SIGNAL_CHUNK_SIZE = 5
_CPP_SIGNAL = (
    10000, 10000, 10000, 10000, 10000, 10, 10, 10, 10, 10,
    10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
    10000, 10000, 10000, 10000, 10000, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10000, 10000, 10000, 10000, 10000,
    10000, 10000, 10000, 10000, 10000, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10000, 10000, 10000, 10000, 10000, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10000, 10000, 10000, 10000, 10000,
)
_CPP_SIGNAL_VAD_RES = (
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
)
_CPP_SIGNAL_LOW_START = (
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10000, 10000, 10000, 10000, 10000,
)
_CPP_SIGNAL_LOW_START_VAD_RES = (1.0, 1.0, 0.0, 0.0, 1.0)

# Constants from tests/{misc_audio_test,vad_patch_creator_test}.cc.
_CPP_CA01_NUM_SAMPLES = 131444
_CPP_CA01_SAMPLE_RATE = 48000
_CPP_CA01_PATCHES = (9, 29, 49, 69, 89)
_FLOAT_WIDTH = re.compile(r"(?:<|x|\b)f(?P<width>16|32|64)\b")
_NP_BOOL_TRUE = jnp.dtype(jnp.bool_).type(True)


def _assert_array_equal(
    actual: jax.typing.ArrayLike, expected: jax.typing.ArrayLike
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim == 0:
    assert bool(jnp.all(a == b))
  else:
    assert a.shape == b.shape and bool(jnp.array_equal(a, b))


@pytest.mark.parametrize(
    ("spectrum_length", "expected"),
    [
        (45, [14]),
        (46, [14]),
        (74, [14]),
        (75, [14]),
        (76, [14, 44]),
        (105, [14, 44]),
        (106, [14, 44, 74]),
    ],
)
def test_create_uniform_patch_indices_boundaries(
    spectrum_length: int, expected: list[int]
):
  got = patches.create_uniform_patch_indices(spectrum_length)
  assert got.dtype == jnp.int32
  _assert_array_equal(got, expected)
  assert int(got[-1]) + patches.AUDIO_PATCH_SIZE <= spectrum_length


def test_create_uniform_patch_indices_custom_size_and_validation():
  _assert_array_equal(
      patches.create_uniform_patch_indices(31, patch_size=21), [9]
  )
  _assert_array_equal(
      patches.create_uniform_patch_indices(52, patch_size=21), [9]
  )
  _assert_array_equal(
      patches.create_uniform_patch_indices(53, patch_size=21), [9, 30]
  )
  for sl, want in ((3, [0]), (4, [0]), (5, [0]), (6, [0, 2])):
    _assert_array_equal(
        patches.create_uniform_patch_indices(sl, patch_size=2), want
    )

  with pytest.raises(ValueError, match="smaller than minimum patch size"):
    patches.create_uniform_patch_indices(44)
  with pytest.raises(ValueError, match="smaller than minimum patch size"):
    patches.create_uniform_patch_indices(30, patch_size=21)
  with pytest.raises(ValueError, match="smaller than minimum patch size"):
    patches.create_uniform_patch_indices(2, patch_size=2)
  for bad_ps in (1, 0, -2, True, _NP_BOOL_TRUE, jnp.array(True), 2.5):
    with pytest.raises(ValueError, match="patch_size must be >= 2"):
      patches.create_uniform_patch_indices(100, patch_size=bad_ps)
  for bad_sl in (-1, True, _NP_BOOL_TRUE, 50.5, 1 << 40):
    with pytest.raises(ValueError, match="spectrum_length"):
      patches.create_uniform_patch_indices(bad_sl)

  # 10000 samples at 48 kHz (tests/short_duration_test.cc) yields 7 frames < 45.
  ws, hop, _ = spectrogram.analysis_window(48000)
  spec_len = int(spectrogram.num_spectrogram_frames(10000, ws, hop))
  assert spec_len == 7
  with pytest.raises(ValueError, match=r"Reference spectrum size \(7\)"):
    patches.create_uniform_patch_indices(spec_len)


def test_rms_vad_matches_cpp_unit_tests_and_hangover_edges():
  chunk_i16 = jnp.asarray(_CPP_CHUNK, dtype=jnp.int16)
  rms = patches.calc_chunk_rms(chunk_i16)
  assert rms.dtype == jnp.float32
  assert float(rms) == pytest.approx(_CPP_CHUNK_RMS, abs=_CPP_VAD_TOLERANCE)
  min_i16 = jnp.array([-32768, -32768], dtype=jnp.int16)
  assert float(patches.calc_chunk_rms(min_i16)) == pytest.approx(32768.0)
  z_grad = jax.grad(lambda c: jnp.sum(patches.calc_chunk_rms(c)))(
      jnp.zeros((2, 4), dtype=jnp.float32)
  )
  _assert_array_equal(z_grad, jnp.zeros((2, 4), dtype=jnp.float32))

  for raw_sig, want in (
      (_CPP_SIGNAL, _CPP_SIGNAL_VAD_RES),
      (_CPP_SIGNAL_LOW_START, _CPP_SIGNAL_LOW_START_VAD_RES),
  ):
    chunks = jnp.asarray(raw_sig, dtype=jnp.int16).reshape(
        -1, _CPP_SIGNAL_CHUNK_SIZE
    )
    got = patches.calc_rms_vad(chunks)
    assert got.dtype == jnp.float32
    _assert_array_equal(got, want)

  # Fewer than 3 chunks are always marked active even when silent.
  for n in (0, 1, 2):
    got = patches.calc_rms_vad(jnp.zeros((n, 4), dtype=jnp.int16))
    assert got.dtype == jnp.float32
    _assert_array_equal(got, jnp.ones((n,), dtype=jnp.float32))

  # Exact threshold: 5000 active, 4999 inactive (3 consecutive 4999s drop to 0).
  levels = jnp.array(
      [[4999, 4999]] * 3 + [[5000, 5000]] + [[4999, 4999]] * 4 + [[5000, 5000]],
      dtype=jnp.int16,
  )
  _assert_array_equal(
      patches.calc_rms_vad(levels),
      [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
  )

  for bad_c in (jnp.zeros((), dtype=jnp.float32), jnp.zeros((2, 0))):
    with pytest.raises(ValueError, match="non-empty last axis"):
      patches.calc_chunk_rms(bad_c)
  for bad_2d in (jnp.zeros((4,), dtype=jnp.float32), jnp.zeros((2, 0))):
    with pytest.raises(ValueError, match="2D array"):
      patches.calc_rms_vad(bad_2d)


def test_get_voice_activity_truncation_clamping_and_cpp_slice_geometry():
  # +-4999.9 truncates toward zero to +-4999 (< 5000), unlike round/floor.
  sig = jnp.concatenate([
      jnp.full((16,), 4999.9 / 32768.0, dtype=jnp.float32),
      jnp.full((4,), 5000.0 / 32768.0, dtype=jnp.float32),
      jnp.full((16,), -4999.9 / 32768.0, dtype=jnp.float32),
      jnp.full((4,), -5000.0 / 32768.0, dtype=jnp.float32),
      jnp.array([2.5, -2.5, 1.0, -1.0], dtype=jnp.float32),
  ])
  _assert_array_equal(
      patches.get_voice_activity(sig, 0, sig.shape[0], 4),
      [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
  )
  # pylint: disable-next=protected-access
  clamped = patches._float_to_int16_samples(
      jnp.array([1.0, 2.0, -1.0, -2.0], dtype=jnp.float32)
  )
  _assert_array_equal(clamped, [32767.0, 32767.0, -32768.0, -32768.0])

  # CleanSpeechVAD slice parameters from tests/vad_patch_creator_test.cc.
  t = jnp.arange(_CPP_CA01_NUM_SAMPLES, dtype=jnp.float32) / jnp.float32(
      _CPP_CA01_SAMPLE_RATE
  )
  synth_speech = jnp.float32(0.4) * jnp.sin(
      jnp.float32(2.0 * math.pi * 220.0) * t
  )
  vad = patches.get_voice_activity(synth_speech, 14, 115200, 480)
  assert vad.shape == (240,) and vad.dtype == jnp.float32

  # Partial trailing frame (total_samples % frame_len != 0) is discarded.
  partial_sig = jnp.concatenate([
      jnp.zeros((20,), dtype=jnp.float32),
      jnp.ones((3,), dtype=jnp.float32),
  ])
  vad_partial = patches.get_voice_activity(partial_sig, 0, 23, 5)
  _assert_array_equal(vad_partial, [1.0, 1.0, 0.0, 0.0])

  for bad_args in (
      (-1, 10, 5),
      (0, 0, 5),
      (0, 10, 0),
      (True, 10, 5),
      (_NP_BOOL_TRUE, 10, 5),
      (0, 10.0, 5),
      (0, 10, 2.5),
  ):
    with pytest.raises(ValueError, match="invalid VAD slice parameters"):
      patches.get_voice_activity(synth_speech, *bad_args)
  with pytest.raises(ValueError, match="smaller than requested VAD range"):
    patches.get_voice_activity(synth_speech, synth_speech.shape[0] - 5, 10, 5)
  for bad_sig in (
      jnp.float32(0.0),
      jnp.zeros((0,), dtype=jnp.float32),
      jnp.ones((20, 1), dtype=jnp.float32),
  ):
    with pytest.raises(ValueError, match="signal.*non-empty 1D"):
      patches.get_voice_activity(bad_sig, 0, 10, 5)


def test_create_vad_patch_indices_cpp_quirks_silence_and_validation():
  # CA01_01 geometry (tests/vad_patch_creator_test.cc): 131444 -> 133 cols.
  ws48, hop48, _ = spectrogram.analysis_window(_CPP_CA01_SAMPLE_RATE)
  spec_len_48 = int(
      spectrogram.num_spectrogram_frames(_CPP_CA01_NUM_SAMPLES, ws48, hop48)
  )
  assert spec_len_48 == 133
  idx_samples = jnp.arange(_CPP_CA01_NUM_SAMPLES, dtype=jnp.int32)
  t48 = idx_samples.astype(jnp.float32) / jnp.float32(_CPP_CA01_SAMPLE_RATE)
  # Active speech across patches 0..4 and trailing silence across patch 5.
  active_until = 9 + 5 * patches.SPEECH_PATCH_SIZE * hop48 - 4 * hop48
  speech_48 = jnp.where(
      idx_samples < active_until,
      jnp.float32(0.5) * jnp.sin(jnp.float32(2.0 * math.pi * 240.0) * t48),
      jnp.float32(0.0),
  )
  idx_pad, num_active = patches.create_vad_patch_indices(
      spec_len_48, speech_48, _CPP_CA01_SAMPLE_RATE
  )
  assert idx_pad.dtype == jnp.int32 and num_active.dtype == jnp.int32
  assert idx_pad.shape == (6,) and int(num_active) == 5
  _assert_array_equal(idx_pad[:5], _CPP_CA01_PATCHES)
  _assert_array_equal(idx_pad[5:], [0])

  # 16 kHz (ws=1280, hop=320), 3 candidate patches -> spec_len = 69.
  ws16, hop16, _ = spectrogram.analysis_window(16000)
  n16 = ws16 + 68 * hop16

  # Signed max=+0.1 (not max(abs)=1.0) keeps patch 2 (0.02/0.1 -> RMS 6553).
  sig_q1 = (
      jnp.zeros((n16,), dtype=jnp.float32)
      .at[0]
      .set(0.1)
      .at[1]
      .set(-1.0)
      .at[9 + 40 * hop16 : 9 + 41 * hop16]
      .set(0.02)
  )
  i1, n1 = patches.create_vad_patch_indices(69, sig_q1, 16000)
  assert int(n1) == 2
  _assert_array_equal(i1, [9, 49, 0])

  # C++ passes frame index first_patch_idx (9) as start_sample, not 9 * hop.
  sig_q2 = (
      jnp.zeros((n16,), dtype=jnp.float32)
      .at[0]
      .set(1.0)
      .at[9 + 20 * hop16 : 9 + 21 * hop16]
      .set(0.5)
  )
  i2, n2 = patches.create_vad_patch_indices(69, sig_q2, 16000)
  assert int(n2) == 2
  _assert_array_equal(i2, [9, 29, 0])

  for pair, want_i16 in (
      ([0.0, 0.0], [32767.0, 32767.0]),
      ([0.0, -0.25], [32767.0, -32768.0]),
      ([-0.5, -0.1], [32767.0, 32767.0]),
  ):
    # Silence (max == 0), non-positive, and strictly negative stay NaN-free.
    arr_pair = jnp.asarray(pair, dtype=jnp.float32)
    # pylint: disable=protected-access
    got_i16 = patches._float_to_int16_samples(
        patches._normalize_signal(arr_pair)
    )
    # pylint: enable=protected-access
    _assert_array_equal(got_i16, want_i16)

  for s_edge in (
      jnp.zeros((n16,), dtype=jnp.float32),
      jnp.concatenate([
          jnp.zeros((1,), dtype=jnp.float32),
          jnp.full((n16 - 1,), -0.25, dtype=jnp.float32),
      ]),
      jnp.full((n16,), -0.25, dtype=jnp.float32),
  ):
    i_e, n_e = patches.create_vad_patch_indices(69, s_edge, 16000)
    assert int(n_e) == 3
    _assert_array_equal(i_e, [9, 29, 49])

  # Exact VAD candidate count boundary: 48 -> 1 patch [9], 49 -> 2 patches.
  ones16 = jnp.ones((n16,), dtype=jnp.float32)
  i48, n48 = patches.create_vad_patch_indices(48, ones16, 16000)
  i49, n49 = patches.create_vad_patch_indices(49, ones16, 16000)
  assert int(n48) == 1 and i48.tolist() == [9]
  assert int(n49) == 2 and i49.tolist() == [9, 29]

  # Non-multiple-of-4 window_size (11025 Hz -> ws=882, hop=220, 882*0.25=220.5).
  ws11, hop11, _ = spectrogram.analysis_window(11025)
  assert (ws11, hop11) == (882, 220)
  sig_11k = jnp.sin(
      jnp.float32(2.0 * math.pi * 440.0 / 11025.0)
      * jnp.arange(11025 * 2, dtype=jnp.float32)
  )
  sl11 = int(
      spectrogram.num_spectrogram_frames(sig_11k.shape[0], ws11, hop11)
  )
  idx11, n11 = patches.create_vad_patch_indices(sl11, sig_11k, 11025)
  assert int(n11) == 4 and idx11.shape == (4,)
  # Exact VAD range bound: 9 + int(20 * (882 * 0.25)) = 4419 (not 4400 or 4420).
  i11_min, n11_min = patches.create_vad_patch_indices(29, sig_11k[:4419], 11025)
  assert int(n11_min) == 1 and int(i11_min[0]) == 9
  with pytest.raises(ValueError, match="smaller than requested VAD range"):
    patches.create_vad_patch_indices(29, sig_11k[:4418], 11025)

  # Exact minimum boundary: spec_len=28 fails, spec_len=29 succeeds.
  sig29 = jnp.ones((ws16 + 28 * hop16,), dtype=jnp.float32)
  with pytest.raises(ValueError, match="smaller than minimum VAD patch size"):
    patches.create_vad_patch_indices(28, sig29, 16000)
  i29, n29 = patches.create_vad_patch_indices(29, sig29, 16000)
  assert int(n29) == 1 and int(i29[0]) == 9
  for bad_ps in (1, True, _NP_BOOL_TRUE, 2.5):
    with pytest.raises(ValueError, match="patch_size must be >= 2"):
      patches.create_vad_patch_indices(40, sig29, 16000, patch_size=bad_ps)
  for bad_sig in (
      jnp.float32(0.0),
      jnp.zeros((0,), dtype=jnp.float32),
      jnp.ones((sig29.shape[0], 1), dtype=jnp.float32),
  ):
    with pytest.raises(ValueError, match="ref_signal.*non-empty 1D"):
      patches.create_vad_patch_indices(29, bad_sig, 16000)
  with pytest.raises(ValueError, match="smaller than requested VAD range"):
    patches.create_vad_patch_indices(100, sig29[:1000], 16000)


def test_create_patches_vad_compaction_jit_vmap_and_validation():
  sr, patch_size = 16000, 20
  ws, hop, _ = spectrogram.analysis_window(sr)
  n_samples = ws + 88 * hop
  spec_len = int(spectrogram.num_spectrogram_frames(n_samples, ws, hop))
  assert spec_len == 89

  spec = jnp.arange(21 * spec_len, dtype=jnp.float32).reshape(21, spec_len)
  empty = patches.create_patches_from_indices(
      spec, jnp.zeros((0,), dtype=jnp.int32), 20
  )
  empty_list = patches.create_patches_from_indices(spec, [], 20)
  assert empty.dtype == empty_list.dtype == jnp.float32
  assert empty.shape == empty_list.shape == (0, 21, 20)
  # Narrow integer indices avoid int8 overflow when max_start > 127.
  wide_spec = jnp.arange(2 * 160, dtype=jnp.float32).reshape(2, 160)
  g_i8 = patches.create_patches_from_indices(
      wide_spec, jnp.array([9], dtype=jnp.int8), 20
  )
  _assert_array_equal(g_i8[0], wide_spec[:, 9:29])

  traces = 0

  @jax.jit
  def extract(sig: jax.Array, sp: jax.Array):
    nonlocal traces
    traces += 1
    idx_pad, n_act = patches.create_vad_patch_indices(
        sp.shape[1], sig, sr, patch_size=patch_size
    )
    return (
        idx_pad,
        n_act,
        patches.create_patches_from_indices(sp, idx_pad, patch_size),
    )

  active_sets = ([0], [0, 2], [0, 1, 3], [0, 1, 2, 3])
  batch_sigs = []
  for active_set in active_sets:
    sig = jnp.zeros((n_samples,), dtype=jnp.float32).at[0].set(1.0)
    for p in active_set:
      if p > 0:
        s0 = 9 + p * patch_size * hop
        sig = sig.at[s0 : s0 + hop].set(0.8)
    batch_sigs.append(sig)
    idx_pad, n_act, gathered = extract(sig, spec)
    want_idx = [9 + p * patch_size for p in active_set]
    assert int(n_act) == len(want_idx) and gathered.shape == (4, 21, patch_size)
    _assert_array_equal(idx_pad[: int(n_act)], want_idx)
    _assert_array_equal(idx_pad[int(n_act) :], 0)
    for k, col in enumerate(want_idx):
      _assert_array_equal(gathered[k], spec[:, col : col + patch_size])
  assert traces == 1

  # vmap across batch with heterogeneous active patch counts.
  b_idx, b_num, b_patches = jax.jit(jax.vmap(lambda s: extract(s, spec)))(
      jnp.stack(batch_sigs)
  )
  _assert_array_equal(b_num, [1, 2, 3, 4])
  assert b_idx.shape == (4, 4) and b_patches.shape == (4, 4, 21, patch_size)

  # Float64 inputs under x64=True coerce to float32, and f32 HLO emits only f32.
  with jax.enable_x64(True):
    idx64, num64 = patches.create_vad_patch_indices(
        spec_len, batch_sigs[1].astype(jnp.float64), sr
    )
    assert idx64.dtype == num64.dtype == jnp.int32 and int(num64) == 2
    g64 = patches.create_patches_from_indices(
        spec.astype(jnp.float64), jnp.array([9, 49], dtype=jnp.int64), 20
    )
    assert g64.dtype == jnp.float32 and g64.shape == (2, 21, 20)
    hlo = extract.lower(
        jax.ShapeDtypeStruct((n_samples,), jnp.float32),
        jax.ShapeDtypeStruct((21, spec_len), jnp.float32),
    ).as_text()
    assert {int(w) for w in _FLOAT_WIDTH.findall(hlo)} == {32}

  with pytest.raises(ValueError, match="non-empty 2D"):
    patches.create_patches_from_indices(jnp.zeros((4,)), jnp.array([0]), 2)
  for bad_ps in (0, -1, True, _NP_BOOL_TRUE, 1.5):
    with pytest.raises(ValueError, match="patch_size must be >= 1"):
      patches.create_patches_from_indices(spec, jnp.array([0]), bad_ps)
  with pytest.raises(ValueError, match="fewer than patch_size"):
    patches.create_patches_from_indices(spec[:, :10], jnp.array([0]), 20)
  with pytest.raises(ValueError, match="1D array"):
    patches.create_patches_from_indices(spec, jnp.array([[0]]), 20)
  for bad_idx in (
      jnp.array([9.5]),
      jnp.array([True]),
      [9, True],
      [9, _NP_BOOL_TRUE],
      [9, jnp.array(True)],
      jnp.zeros((0,), dtype=jnp.float32),
      jnp.zeros((0,), dtype=jnp.bool_),
      jnp.zeros((0,), dtype=jnp.complex64),
  ):
    with pytest.raises(ValueError, match="integer dtype"):
      patches.create_patches_from_indices(spec, bad_idx, 20)
  for oob_idx in (
      jnp.array([-1]),
      jnp.array([spec_len - patch_size + 1]),
      [1 << 40],
      [10**30],
  ):
    with pytest.raises(ValueError, match=r"patch_indices must be in \[0,"):
      patches.create_patches_from_indices(spec, oob_idx, patch_size)
