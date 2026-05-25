import io
import logging

import numpy as np
import torch
import librosa
import nltk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from g2p_en import G2p

from app.config import get_settings
from app.common import (
    ctc_decode_token_ids,
    normalize_phonemes,
    phonemes_to_ipa,
    phonemes_to_ipa_tokens,
    preprocess_audio,
    save_debug_audio_preview,
    strip_g2p_tokens,
    compute_score,
)

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


def predict_phonemes(audio_bytes: bytes) -> list[str]:
    """Audio bytes → list of predicted ARPAbet phonemes."""
    _load_model()

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=settings.SAMPLE_RATE, mono=True)
    max_len = int(settings.MAX_AUDIO_DURATION_SEC * settings.SAMPLE_RATE)
    audio_before = audio[:max_len].astype(np.float32)

    audio_after = preprocess_audio(audio_before.copy())

    if settings.DEBUG_SAVE_AUDIO:
        save_debug_audio_preview(
            audio_before,
            audio_after,
            settings.SAMPLE_RATE,
            settings.DEBUG_AUDIO_DIR,
            play_audio=settings.DEBUG_PLAY_AUDIO,
        )

    inputs = processor(
        audio_after, sampling_rate=settings.SAMPLE_RATE, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    return ctc_decode_token_ids(processor.tokenizer, predicted_ids)


def expected_phonemes(text: str) -> list[str]:
    """English text → list of expected ARPAbet phonemes via g2p_en.

    Output is normalized to the same 39-phoneme set the model was trained on.
    """
    _load_model()
    raw = g2p(text)
    return normalize_phonemes(strip_g2p_tokens(raw))


__all__ = [
    "_load_model",
    "predict_phonemes",
    "expected_phonemes",
    "compute_score",
    "phonemes_to_ipa",
    "phonemes_to_ipa_tokens",
]
