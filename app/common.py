import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Phoneme constants ────────────────────────────────────────────────────────

STRESS_RE = re.compile(r"[012]$")

# TimitBet 61 → 39 phoneme mapping
# Lee, K.-F., & Hon, H.-W. (1989). IEEE TASSP, 37(11), 1641–1648.
PHON61_TO_39: dict[str, str] = {
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
SILENCE = frozenset({'h#'})

# ARPAbet (39-phoneme set) → IPA
ARPABET_TO_IPA: dict[str, str] = {
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

SKIP_TOKENS = frozenset({
    '|', '<pad>', '<s>', '</s>', '<unk>',
    '[pad]', '[unk]', '[sep]', '[cls]', '[mask]',
})


# ── Phoneme utilities ────────────────────────────────────────────────────────

def phonemes_to_ipa(phonemes: list[str]) -> str:
    """Convert a list of ARPAbet phonemes to a space-joined IPA string."""
    return " ".join(ARPABET_TO_IPA.get(p, p) for p in phonemes)


def normalize_phonemes(tokens: list[str]) -> list[str]:
    """Apply phon61→39 mapping and drop silence markers."""
    result = []
    for tok in tokens:
        mapped = PHON61_TO_39.get(tok, tok)
        if mapped not in SILENCE:
            result.append(mapped)
    return result


def strip_g2p_tokens(raw: list[str]) -> list[str]:
    """Strip stress digits from g2p tokens; keep only alphabetic tokens."""
    return [STRESS_RE.sub("", tok).lower() for tok in raw if re.search(r"[a-zA-Z]", tok)]


def ctc_decode_tokens(tokenizer, ids: list[int], blank_token: str) -> list[str]:
    """CTC greedy decode: remove blanks, word-boundary markers, and consecutive duplicates."""
    tokens = tokenizer.convert_ids_to_tokens(ids)
    blank_token_low = blank_token.lower()
    decoded: list[str] = []
    prev: str | None = None
    for tok in tokens:
        tok_low = tok.lower()
        if tok_low == blank_token_low or tok_low in SKIP_TOKENS:
            prev = None
            continue
        if tok_low != prev:
            decoded.append(tok_low)
            prev = tok_low
    return decoded


def save_debug_audio_preview(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sample_rate: int,
    output_dir: str,
    *,
    play_audio: bool = False,
) -> tuple[Path, Path]:
    """Lưu WAV trước/sau preprocess; in đường dẫn ra terminal để nghe thủ công."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path_before = out / f"{stamp}_before_preprocess.wav"
    path_after = out / f"{stamp}_after_preprocess.wav"

    sf.write(path_before, audio_before.astype(np.float32), sample_rate)
    sf.write(path_after, audio_after.astype(np.float32), sample_rate)

    logger.info("─── Debug audio (trước preprocess) ───")
    logger.info("  %s", path_before.resolve())
    logger.info("─── Debug audio (sau preprocess, đưa vào model) ───")
    logger.info("  %s", path_after.resolve())

    if play_audio:
        _play_debug_wavs(path_before, path_after)

    return path_before, path_after


def _play_debug_wavs(*paths: Path) -> None:
    try:
        import sounddevice as sd

        for path in paths:
            data, sr = sf.read(path, dtype="float32")
            logger.info("Đang phát: %s", path.name)
            sd.play(data, sr)
            sd.wait()
    except ImportError:
        for path in paths:
            logger.info("Mở file: %s", path.name)
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')


def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """Preprocess raw audio signal before phoneme inference.

    Steps:
    1. Trim leading/trailing silence (top_db=30 dB threshold).
    2. Pre-emphasis filter  y[n] = y[n] - 0.97 * y[n-1]  to boost
       high-frequency speech components.
    3. Peak-normalize so the loudest sample is ±1.0.
    """
    audio, _ = librosa.effects.trim(audio, top_db=30)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


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
    per = dp[m] / n
    return round(max(0.0, (1.0 - per) * 100.0), 2)
