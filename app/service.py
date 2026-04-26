import io
import re
import logging

import numpy as np
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

# TimitBet 61 → 39 phoneme mapping
# Lee, K.-F., & Hon, H.-W. (1989). IEEE TASSP, 37(11), 1641–1648.
_PHON61_TO_39: dict[str, str] = {
    'iy': 'iy',  'ih': 'ih',  'eh': 'eh',   'ae': 'ae',  'ix': 'ih',  'ax': 'ah',
    'ah': 'ah',  'uw': 'uw',  'ux': 'uw',   'uh': 'uh',  'ao': 'aa',  'aa': 'aa',
    'ey': 'ey',  'ay': 'ay',  'oy': 'oy',   'aw': 'aw',  'ow': 'ow',
    'l':  'l',   'el': 'l',   'r':  'r',    'y':  'y',   'w':  'w',
    'er': 'er',  'axr':'er',  'm':  'm',    'em': 'm',   'n':  'n',
    'nx': 'n',   'en': 'n',   'ng': 'ng',   'eng':'ng',  'ch': 'ch',
    'jh': 'jh',  'dh': 'dh',  'b':  'b',    'd':  'd',   'dx': 'd',
    'g':  'g',   'p':  'p',   't':  't',    'k':  'k',   'z':  'z',
    'zh': 'sh',  'v':  'v',   'f':  'f',    'th': 'th',  's':  's',
    'sh': 'sh',  'hh': 'hh',  'hv': 'hh',
    # silence / boundary tokens → filtered out
    'pcl':'h#',  'tcl':'h#',  'kcl':'h#',   'qcl':'h#',  'bcl':'h#',
    'dcl':'h#',  'gcl':'h#',  'h#': 'h#',   '#h': 'h#',  'pau':'h#',
    'epi':'h#',  'ax-h':'ah', 'q':  'h#',
}
_SILENCE = frozenset({'h#'})

# ARPAbet (39-phoneme set) → IPA
_ARPABET_TO_IPA: dict[str, str] = {
    'iy': 'iː',  'ih': 'ɪ',   'eh': 'ɛ',   'ae': 'æ',
    'ah': 'ʌ',   'uw': 'uː',  'uh': 'ʊ',   'aa': 'ɑː',
    'ey': 'eɪ',  'ay': 'aɪ',  'oy': 'ɔɪ',  'aw': 'aʊ',
    'ow': 'oʊ',  'er': 'ɜr',
    'l':  'l',   'r':  'r',   'y':  'j',   'w':  'w',
    'm':  'm',   'n':  'n',   'ng': 'ŋ',
    'ch': 'tʃ',  'jh': 'dʒ',
    'dh': 'ð',   'v':  'v',   'f':  'f',   'th': 'θ',
    's':  's',   'sh': 'ʃ',   'hh': 'h',   'z':  'z',
    'b':  'b',   'd':  'd',   'g':  'ɡ',   'p':  'p',
    't':  't',   'k':  'k',
}


def phonemes_to_ipa(phonemes: list[str]) -> str:
    """Convert a list of ARPAbet phonemes to a space-joined IPA string."""
    return " ".join(_ARPABET_TO_IPA.get(p, p) for p in phonemes)


_SKIP_TOKENS = frozenset({'|', '<pad>', '<s>', '</s>', '<unk>'})


def _ctc_decode_tokens(ids: list[int], blank_token: str) -> list[str]:
    """CTC greedy decode: remove blanks, word-boundary markers, and consecutive duplicates."""
    tokens = processor.tokenizer.convert_ids_to_tokens(ids)
    decoded: list[str] = []
    prev: str | None = None
    for tok in tokens:
        tok_low = tok.lower()
        if tok == blank_token or tok_low in _SKIP_TOKENS:
            prev = None
            continue
        if tok_low != prev:
            decoded.append(tok_low)
            prev = tok_low
    return decoded


def _normalize_predicted(tokens: list[str]) -> list[str]:
    """Apply phon61→39 mapping and drop silence markers."""
    result = []
    for tok in tokens:
        mapped = _PHON61_TO_39.get(tok, tok)
        if mapped not in _SILENCE:
            result.append(mapped)
    return result


def _preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """Preprocess raw audio signal before phoneme inference.

    Steps:
    1. Trim leading/trailing silence (top_db=30 dB threshold).
    2. Pre-emphasis filter  y[n] = y[n] - 0.97 * y[n-1]  to boost
       high-frequency speech components.
    3. Peak-normalize so the loudest sample is ±1.0.
    """
    # 1. Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # 2. Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)

    # 3. Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


def predict_phonemes(audio_bytes: bytes) -> list[str]:
    """Audio bytes → list of predicted ARPAbet phonemes."""
    _load_model()

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=settings.SAMPLE_RATE, mono=True)
    max_len = int(settings.MAX_AUDIO_DURATION_SEC * settings.SAMPLE_RATE)
    audio = audio[:max_len]

    # audio = _preprocess_audio(audio)

    inputs = processor(audio, sampling_rate=settings.SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    ids = torch.argmax(logits, dim=-1)[0].tolist()
    blank_token = processor.tokenizer.pad_token  # CTC blank = <pad>
    tokens = _ctc_decode_tokens(ids, blank_token)
    result = _normalize_predicted(tokens)
    return result


def expected_phonemes(text: str) -> list[str]:
    """English text → list of expected ARPAbet phonemes via g2p_en.

    Output is normalized to the same 39-phoneme set the model was trained on:
    stress digits are stripped, then each token is mapped through _PHON61_TO_39
    and silence markers are dropped — identical to what _normalize_predicted does
    for the predicted side.
    """
    _load_model()
    raw = g2p(text)
    # Strip stress digits and lowercase, keep only alphabetic tokens
    stripped = [_STRESS_RE.sub("", tok).lower() for tok in raw if re.search(r"[a-zA-Z]", tok)]
    # Collapse 61-phoneme variants → 39-phoneme set and drop silence
    result = []
    for tok in stripped:
        mapped = _PHON61_TO_39.get(tok, tok)
        if mapped not in _SILENCE:
            result.append(mapped)
    return result


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
