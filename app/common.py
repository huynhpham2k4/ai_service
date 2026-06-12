import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import Levenshtein
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Phoneme constants ────────────────────────────────────────────────────────

STRESS_RE = re.compile(r"[012]$")

# TimitBet 61 → 39 phoneme mapping
# Lee, K.-F., & Hon, H.-W. (1989). IEEE TASSP, 37(11), 1641–1648.
PHON61_TO_39: dict[str, str] = {
    'iy': 'iy',  'ih': 'ih',   'eh': 'eh',  'ae': 'ae',    'ix': 'ih',  'ax': 'ah',   'ah': 'ah',  'uw': 'uw',
    'ux': 'uw',  'uh': 'uh',   'ao': 'aa',  'aa': 'aa',    'ey': 'ey',  'ay': 'ay',   'oy': 'oy',  'aw': 'aw',
    'ow': 'ow',  'l': 'l',     'el': 'l',   'r': 'r',      'y': 'y',    'w': 'w',     'er': 'er',  'axr': 'er',
    'm': 'm',    'em': 'm',    'n': 'n',    'nx': 'n',     'en': 'n',   'ng': 'ng',   'eng': 'ng', 'ch': 'ch',
    'jh': 'jh',  'dh': 'dh',   'b': 'b',    'd': 'd',      'dx': 'd',  'g': 'g',     'p': 'p',    't': 't',
    'k': 'k',    'z': 'z',     'zh': 'sh',  'v': 'v',      'f': 'f',    'th': 'th',   's': 's',    'sh': 'sh',
    'hh': 'hh',  'hv': 'hh',   'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'qcl': 'h#', 'bcl': 'h#', 'dcl': 'h#',
    'gcl': 'h#', 'h#': 'h#',   '#h': 'h#',  'pau': 'h#', 'epi': 'h#', 'ax-h': 'ah', 'q': 'h#',
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

# CTC special tokens → loại khỏi argmax trước khi decode (thêm token mới tại đây)
CTC_SPECIAL_TOKENS: tuple[str, ...] = (
    "[PAD]",
    " ",
    "h#",
    '|'
)


# ── Phoneme utilities ────────────────────────────────────────────────────────

def phonemes_to_ipa(phonemes: list[str]) -> str:
    """Convert a list of ARPAbet phonemes to a space-joined IPA string."""
    return " ".join(ARPABET_TO_IPA.get(p, p) for p in phonemes)


def phonemes_to_ipa_tokens(phonemes: list[str]) -> list[str]:
    """Convert a list of ARPAbet phonemes to a list of IPA tokens."""
    return [ARPABET_TO_IPA.get(p, p) for p in phonemes]


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


def get_ctc_special_token_ids(tokenizer) -> frozenset[int]:
    """Chuyển CTC_SPECIAL_TOKENS → id; dùng để loại khỏi token_ids trước khi decode phonemes."""
    vocab = tokenizer.get_vocab()
    ids: set[int] = set()
    for token in CTC_SPECIAL_TOKENS:
        if token in vocab:
            ids.add(tokenizer.encode(token, add_special_tokens=False)[0])
    return frozenset(ids)


def collapse_tokens(tokens: list) -> list:
    """CTC collapse: bỏ token_id lặp liên tiếp (giữ 1 trong chuỗi trùng)."""
    prev_token = None
    out: list = []
    for token in tokens:
        if token != prev_token and prev_token is not None:
            out.append(prev_token)
        prev_token = token
    if prev_token is not None:
        out.append(prev_token)
    return out


def clean_token_ids(token_ids: list[int], special_ids: frozenset[int]) -> list[int]:
    """Remove special token ids, then collapse duplicated token ids."""
    filtered = [x for x in token_ids if x not in special_ids]
    return collapse_tokens(filtered)


def ctc_decode_token_ids(tokenizer, token_ids: list[int]) -> list[str]:
    """clean_token_ids → mỗi id = 1 phoneme trong vocab."""
    special_ids = get_ctc_special_token_ids(tokenizer)
    cleaned = clean_token_ids(token_ids, special_ids)
    phonemes: list[str] = []
    for tok in tokenizer.convert_ids_to_tokens(cleaned):
        p = tok.strip().lower()
        phonemes.append(p)
    return phonemes


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
    """Phoneme accuracy score 0–100 via Levenshtein.editops().

    PER   = len(editops(predicted → expected)) / len(expected)
    Score = (1 - PER) * 100, clamped to [0, 100].
    """
    if not expected:
        return 100.0 if not predicted else 0.0
    ops = Levenshtein.editops(predicted, expected)
    per = len(ops) / len(expected)
    return round(max(0.0, (1.0 - per) * 100.0), 2)


def align_and_trim_noise(expected: list[str], predicted: list[str]) -> dict:
    """Align expected and predicted IPA phoneme sequences, filter out all insertions in predicted
    (where expected is '-'), and recalculate score, PER, and edit distance.
    
    Inputs expected and predicted are list of IPA tokens.
    """
    if not expected:
        return {
            "normalized_predicted": predicted,
            "aligned_expected": [],
            "aligned_predicted": [],
            "distance": len(predicted),
            "score": 0.0 if predicted else 100.0,
        }
        
    if not predicted:
        return {
            "normalized_predicted": [],
            "aligned_expected": expected,
            "aligned_predicted": ["-"] * len(expected),
            "distance": len(expected),
            "score": 0.0,
        }

    # 1. Perform Levenshtein alignment between expected and predicted
    opcodes = Levenshtein.opcodes(expected, predicted)
    aligned_expected = []
    aligned_predicted = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(i2 - i1):
                aligned_expected.append(expected[i1 + k])
                aligned_predicted.append(predicted[j1 + k])
        elif tag == 'replace':
            len_e = i2 - i1
            len_p = j2 - j1
            min_len = min(len_e, len_p)
            for k in range(min_len):
                aligned_expected.append(expected[i1 + k])
                aligned_predicted.append(predicted[j1 + k])
            if len_e > len_p:
                for k in range(min_len, len_e):
                    aligned_expected.append(expected[i1 + k])
                    aligned_predicted.append('-')
            # Insertions (len_p > len_e leftovers) are ignored directly
        elif tag == 'delete':
            for k in range(i2 - i1):
                aligned_expected.append(expected[i1 + k])
                aligned_predicted.append('-')
        # Insertions (tag == 'insert') are ignored directly

    # The normalized predicted is the predicted sequence with all insertions removed
    normalized_predicted = [p for p in aligned_predicted if p != '-']

    # 2. Compute edit distance and score (ignoring insertions)
    distance = sum(1 for e, p in zip(aligned_expected, aligned_predicted) if e != p)
    per = distance / len(expected)
    score = round(max(0.0, (1.0 - per) * 100.0), 2)

    return {
        "normalized_predicted": normalized_predicted,
        "aligned_expected": aligned_expected,
        "aligned_predicted": aligned_predicted,
        "distance": distance,
        "score": score,
    }

