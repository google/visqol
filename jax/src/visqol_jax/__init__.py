"""ViSQOL v3.3.3 (speech and audio modes) in JAX."""

from visqol_jax.io import load_wav_as_mono
from visqol_jax.io import visqol_files
from visqol_jax.selector import PatchSimilarityResult
from visqol_jax.visqol import visqol
from visqol_jax.visqol import visqol_audio
from visqol_jax.visqol import visqol_speech
from visqol_jax.visqol import VisqolOptions
from visqol_jax.visqol import VisqolResult

__version__ = "0.0.1"

__all__ = [
    "PatchSimilarityResult",
    "VisqolOptions",
    "VisqolResult",
    "__version__",
    "load_wav_as_mono",
    "visqol",
    "visqol_audio",
    "visqol_files",
    "visqol_speech",
]
