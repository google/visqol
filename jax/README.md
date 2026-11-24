# visqol-jax

ViSQOL v3.3.3 (speech and audio modes) in pure Python and JAX.

`visqol-jax` ports [google/visqol](https://github.com/google/visqol) (`v3.3.3`)
without TensorFlow, TFLite, libsvm, Armadillo, PFFFT, or C++ runtime
dependencies. Only `jax` and `numpy`
(`dependencies = ["jax>=0.5", "numpy>=1.26"]`) and the Python standard library
are required at runtime, default `float32` execution matches C++ `v3.3.3` within
`< 1e-2` MOS-LQO delta (median `1.5e-4`), and warm `jax.jit` CPU execution runs
3x-5.5x faster than `-c opt` C++ ViSQOL on conformance clips.

> [!IMPORTANT]
> `visqol-jax` reproduces `google/visqol` `v3.3.3` numerically in `float32`,
> including the upstream behaviors listed under
> [Preserved C++ v3.3.3 behaviors](#preserved-c-v333-behaviors).

## Install

Requires Python `>=3.12` (`py3-none-any` wheel,
`dependencies = ["jax>=0.5", "numpy>=1.26"]`, with no `jaxlib` pin so
`jax[cuda12]` and `jax[tpu]` environments install cleanly):

```bash
pip install "git+https://github.com/google/visqol.git#subdirectory=jax"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add "visqol-jax @ git+https://github.com/google/visqol.git#subdirectory=jax"
```

From a local `visqol` repository checkout:

```bash
pip install ./jax
```

## API Usage

```python
import jax
import visqol_jax
from visqol_jax import VisqolOptions

# Speech mode (default 16 kHz, 21 ERB bands 50-8000 Hz, RMS-VAD, lattice MOS).
speech = visqol_jax.visqol_speech(reference, degraded, sample_rate=16000)
mos_speech = speech.moslqo

# Audio mode (48 kHz, 32 ERB bands 50 Hz to Nyquist, uniform patches, nu-SVR).
audio = visqol_jax.visqol_audio(reference_48k, degraded_48k, sample_rate=48000)
mos_audio = audio.moslqo

# Full options control via visqol() and VisqolOptions.
opts = VisqolOptions(
    use_speech_scoring=True,
    use_lattice_model=False,  # TCD-VOIP exponential mapping
    use_unscaled_speech_mos_mapping=False,
    search_window_radius=60,
    skip_mos_mapping=False,
    disable_global_alignment=False,
    disable_realignment=False,
    svr_model_path=None,  # .txt/.txt.gz SVR or .npz lattice archive
)
result = visqol_jax.visqol(reference, degraded, sample_rate=16000, options=opts)
```

Inputs are coerced once at public entry points via `jnp.asarray(...,
dtype=jnp.float32)` and execute in default `float32` JAX without extra JAX
configuration.

`VisqolResult` is registered as a JAX pytree whose floating-point leaves have
dtype `float32`: `moslqo`, `vnsim`, `fvnsim`, `fvnsim10`, `fstdnsim`,
`fvdegenergy`, `alignment_lag_s`, `center_freq_bands`, and `patch_sims`.
`patch_sims` is a `PatchSimilarityResult` pytree containing per-patch
`similarity`, `freq_band_means`, `freq_band_stddevs`, `freq_band_deg_energy`,
`ref_patch_start_time`, `ref_patch_end_time`, `deg_patch_start_time`,
`deg_patch_end_time`, and `int32` `num_patches` (active matched patches occupy
`[:num_patches]`, with trailing static slots zeroed). Both `visqol_speech` and
`visqol_audio` compose inside outer `jax.jit` and `jax.vmap` for batched
scoring:

```python
batch_score = jax.jit(
    jax.vmap(
        lambda r, d: visqol_jax.visqol_speech(r, d, sample_rate=16000).moslqo
    )
)
scores = batch_score(ref_batch, deg_batch)
```

Custom mapper models can be supplied via `svr_model_path` on `VisqolOptions`,
`visqol_speech`, or `visqol_audio`:

- Audio mode accepts a libsvm RBF `.txt` or `.txt.gz` SVR model.
- Speech lattice mode (`use_speech_scoring=True`, `use_lattice_model=True`)
  accepts a `.npz` lattice archive matching `LatticeModel` fields. Convert a
  TensorFlow Lattice `.tflite` model to `.npz` with
  `scripts/export_lattice_npz.py` (`uv run --group convert python
  scripts/export_lattice_npz.py IN.tflite OUT.npz`).
- Speech exponential mode (`use_lattice_model=False`) and
  `skip_mos_mapping=True` ignore `svr_model_path`.

Validation warnings matching C++ `VisqolManager::ValidateInputAudio` (duration
mismatch `> 1 s`, speech sample rate `> 16 kHz`, audio sample rate `!= 48 kHz`)
are emitted via Python `logging`.

### WAV I/O and CLI

Pure-stdlib/JAX 16-bit WAV loading (PCM `0x0001` and
`WAVE_FORMAT_EXTENSIBLE` `0xfffe`, decoded via `array.array("h")` and
`jnp.frombuffer(..., dtype=jnp.int16)` into `float32` `samples / 32768.0f` and
multi-channel downmixed to mono matching `MiscAudio::ToMono`) and the
`visqol-jax` CLI are included:

```python
import visqol_jax
from visqol_jax import VisqolOptions

samples, sr = visqol_jax.load_wav_as_mono("ref.wav")  # 1D float32 jax.Array

res = visqol_jax.visqol_files(
    "ref.wav", "deg.wav", VisqolOptions(use_speech_scoring=True)
)
```

```bash
# Single pair (speech mode, verbose FVNSIM and per-patch tables)
visqol-jax --reference_file ref.wav --degraded_file deg.wav --use_speech_mode --verbose

# Batch CSV with C++-compatible CSV and proto3 debug JSON outputs
visqol-jax --batch_input_csv pairs.csv --results_csv out.csv --output_debug debug.json
```

Supported CLI flags match C++ ViSQOL v3.3.3 (`--reference_file`,
`--degraded_file`, `--batch_input_csv`, `--results_csv`, `--output_debug`,
`--similarity_to_quality_model`, `--search_window_radius`, `--use_speech_mode`,
`--use_lattice_model`/`--nouse_lattice_model`,
`--use_unscaled_speech_mos_mapping`, `--skip_mos_mapping`,
`--disable_global_alignment`, `--disable_realignment`, `--verbose`).

> [!NOTE]
> ViSQOL uses argmax envelope lag alignment, integer sample truncation, RMS-VAD
> gating, and dynamic-programming patch selection, which have zero derivative
> almost everywhere; use `visqol-jax` as a forward evaluation metric. When
> global alignment leaves zero valid degraded spectrogram frames or zero matched
> patches, `visqol` returns `NaN` score fields and zeroed `patch_sims`
> (`num_patches == 0`) while keeping `alignment_lag_s` and `center_freq_bands`
> finite; `visqol_files` raises `ValueError` on unscorable pairs to match C++.
> Matching C++ `ScaleToMatchSoundPressureLevel`, an all-zero `reference` paired
> with a non-zero `degraded` scales `degraded` to `0.0` (`10**(-inf/20) = 0`)
> and returns `vnsim = 1.0`.

## Default Float32 Parity

All stages execute in default `float32` JAX without mutating `jax_enable_x64`,
emitting zero `float64` or `complex128` `stablehlo.fft` ops even when
`jax_enable_x64=True`:

- Slaney ERB filter coefficients and gammatone impulse responses are synthesized
  in Python `float64` (`math` / `cmath`) and cast once to `float32`, matching
  C++ filter design while running FFT-based filterbank convolution in `float32`.
- Local `3x3` patch centering in `nsim` and centered pooled between-patch
  variance in `visqol` avoid catastrophic `float32` sum-of-squares cancellation.
- `lattice`, `mappers`, and `svr` evaluate the 60-lattice
  `CalibratedLatticeEnsemble`, exponential speech curve, and 317-SV audio RBF
  nu-SVR in `float32`.

Measured differential parity in default `float32` JAX against C++
`google/visqol` `v3.3.3` across speech and audio conformance and edge-case
clips:

| Metric | Tolerance / Measured Delta vs C++ | Notes |
|---|---:|---|
| `moslqo` (all conformance and edge cases) | `< 1e-2` (median `1.5e-4`) | Finite `float32` outputs across speech lattice, speech exponential, and audio nu-SVR |
| `moslqo` (audio nu-SVR conformance clips) | `< 2e-3` (`4.37e-4` max) | Tested across `conformance_testdata_subset` and 5 s cropped clips |
| `vnsim` | `< 1e-2` (`< 2e-3` audio conformance) | Mean across active ERB frequency bands |
| `alignment_lag_s` / `center_freq_bands` | Exact sample lag / `rtol <= 1e-6` | Exact sample lag and `float32`-rounded seconds lag |

## Performance

All pipeline stages compile with `jax.jit` over fixed-dimension buffers:
alignment lags, degraded lengths, RMS-VAD patch counts, and fine-realigned
slices use masked and compacted arrays rather than data-dependent crops.
Same-length clips reuse compiled XLA executables regardless of lag, silence, or
active VAD patch count.

Measured on Linux x86_64 CPU against `-c opt` C++ `google/visqol` `v3.3.3`
(batch steady-state time per pair via `--batch_input_csv` to exclude process
startup and model loading):

| Clip (mode, duration, sample rate) | JAX f32 cold | JAX f32 warm | C++ `-c opt` (steady) | f32 speedup |
|---|---:|---:|---:|---:|
| `CA01_01` transcoded (speech lattice, 2.74 s, 48 kHz) | 2.82 s | 0.47 s | 1.38 s | 3.0x |
| `alignment` ref/deg (speech lattice, 20.00 s, 48 kHz) | 5.84 s | 2.73 s | 13.71 s | 5.0x |
| `castanets` identity (audio nu-SVR, 7.28 s, 48 kHz) | 3.54 s | 1.55 s | 6.49 s | 4.2x |
| `guitar` 64 kbps AAC (audio nu-SVR, 12.45 s, 48 kHz) | 4.61 s | 2.49 s | 13.74 s | 5.5x |
| Synthetic 16 kHz speech (speech lattice, 2.00 s, 16 kHz) | 2.18 s | 0.17 s | 0.28 s | 1.7x |

Set `jax_compilation_cache_dir` to persist compiled executables across
processes.

## Package Layout

`src/visqol_jax/_model_data/` bundles `svr_model.txt.gz` (~51 KiB
gzip-compressed libsvm text model) and `speech_lattice.npz` (~550 KiB
compressed NumPy archive generated from upstream `model/*.tflite` via
`scripts/export_lattice_npz.py`), loaded via stdlib `importlib.resources`,
`gzip`, and `numpy.load(..., allow_pickle=False)`. Unit and parity tests
(`*_test.py`) live under `tests/` (without `__init__.py`) and are excluded from
built wheels; `scripts/` is excluded from built wheels as well.

## Preserved C++ v3.3.3 Behaviors

| Upstream source | Preserved behavior |
|---|---|
| `vad_patch_creator.cc` | Passes frame index `first_patch_idx` (`patch_size // 2 - 1 = 9`) directly as `start_sample` to `GetVoiceActivity` instead of `first_patch_idx * frame_size`, starting the VAD window at sample 9. |
| `misc_math.cc` | `MiscMath::Normalize` divides by the signed maximum (`*std::max_element`) rather than `max(abs(x))`; all-zero and non-positive signals follow C++ `int16` clamp semantics without producing `NaN`. |
| `spectrogram.cc` | `Spectrogram::ConvertToDb` computes `10 * log10(\|x\|)` (power dB) on RMS filterbank amplitudes rather than `20 * log10(\|x\|)`, replacing exact zeros with `float32(2.220446e-16)` before the `-45` dB floor. |
| `comparison_patches_selector.cc` | `Slice` computes `end_index = min(N - 1, trunc(end_time * fs))` and slices `[start_index, end_index - 1]` (dropping the final sample when `end_time * fs >= N`), prepends `postsilence` before the audio ahead of `presilence`, and reverts fine realignment to the coarse match only on strict `new_sim < old_sim` (adopting the realigned patch on ties). |
| `visqol.cc` | `CalcFrameDuration` truncates `window.size * window.overlap` to `size_t` before dividing by `sample_rate`; `fvnsim10` averages the lowest `max(1, int(num_patches * 0.10))` patch band means; `fstdnsim` pools patch standard deviations using `ceil((ref_end - ref_start) / frame_duration)` frame counts; `AlterForSimilarityExtremes` clamps `moslqo` to `1.0` whenever `vnsim < 0.15`. |
| `analysis_window.cc` | Uses a symmetric Hann window (`0.5 - 0.5 * cos(2 * pi * i / (N - 1))`) and half-away-from-zero `std::round(sample_rate * window_duration)` sizing. |
| `envelope.cc` + `fast_fourier_transform.cc` | `Envelope::Hilbert` applies `PFFFT_REAL` `float32` inverse FFT directly to the scaled one-sided spectrum (leaving the Nyquist bin zero unless `N == fft_len`) and takes `abs(irfft) + mean`. |
| `speech_similarity_to_quality_mapper.cc` | Casts `kFitScale = 1.245063` through single-precision `float` before multiplying the exponential MOS fit. |

## Development

From the `jax/` directory:

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check .
```

To verify wheel packaging in isolation:

```bash
uv build --wheel --out-dir /tmp/visqol-jax-dist .
CI=1 uv run --isolated --no-project --with /tmp/visqol-jax-dist/*.whl --with pytest pytest tests/packaging_test.py
uv run --isolated --no-project --with /tmp/visqol-jax-dist/*.whl visqol-jax --help
```

## License

Apache-2.0; see
[LICENSE](https://github.com/google/visqol/blob/master/LICENSE).
