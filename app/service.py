import io
import re
import logging

import torch
import librosa
import nltk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from g2p_en import G2p

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ── Globals (lazy-loaded) ────────────────────────────────────────────────────

processor: Wav2Vec2Processor | None = None
model: Wav2Vec2ForCTC | None = None
g2p: G2p | None = None


def _load_model():
    """Load model + processor + g2p once on first call."""
    global processor, model, g2p
    if processor is not None:
        return
    logger.info("Loading model %s ...", settings.MODEL_NAME)
    processor = Wav2Vec2Processor.from_pretrained(
        settings.MODEL_NAME, cache_dir=settings.MODEL_CACHE_DIR
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        settings.MODEL_NAME, cache_dir=settings.MODEL_CACHE_DIR
    )
    model.eval()
    g2p = G2p()
    logger.info("Model loaded.")


# ── Core functions ───────────────────────────────────────────────────────────

_STRESS_RE = re.compile(r"[012]$")


def predict_phonemes(audio_bytes: bytes) -> list[str]:
    """Audio bytes → list of predicted ARPAbet phonemes."""
    _load_model()

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=settings.SAMPLE_RATE, mono=True)
    max_len = int(settings.MAX_AUDIO_DURATION_SEC * settings.SAMPLE_RATE)
    audio = audio[:max_len]

    inputs = processor(audio, sampling_rate=settings.SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    ids = torch.argmax(logits, dim=-1)
    raw = processor.batch_decode(ids)[0]
    return [t.lower() for t in raw.replace("|", " ").split() if t.strip()]


def expected_phonemes(text: str) -> list[str]:
    """English text → list of expected ARPAbet phonemes via g2p_en."""
    _load_model()
    raw = g2p(text)
    return [_STRESS_RE.sub("", tok).lower() for tok in raw if re.search(r"[a-zA-Z]", tok)]


def compute_score(expected: list[str], predicted: list[str]) -> float:
    """Phoneme accuracy score 0-100 based on Levenshtein distance."""
    if not expected:
        return 100.0 if not predicted else 0.0
    n, m = len(expected), len(predicted)
    dp = list(range(m + 1))
    for i in range(n):
        new_dp = [i + 1]
        for j in range(m):
            cost = 0 if expected[i] == predicted[j] else 1
            new_dp.append(min(dp[j] + cost, dp[j + 1] + 1, new_dp[-1] + 1))
        dp = new_dp
    per = dp[m] / n  # phoneme error rate
    return round(max(0.0, (1.0 - per) * 100.0), 2)
