"""
Crawl pronunciation audio from Free Dictionary API for CMU Dictionary words.

Why Free Dictionary API instead of edge-tts:
  - Audio recorded by real native speakers (Wikimedia Commons), not TTS synthesis.
  - A 404 response naturally filters out non-dictionary tokens (punctuation,
    abbreviations, proper-noun-only tokens) without any hand-crafted rules.
  - Completely free, no API key, stable CDN-hosted MP3 files.
  - Prioritises US English (-us) audio, falls back to UK (-uk) then any variant.

API endpoint: https://api.dictionaryapi.dev/api/v2/entries/en/{word}
"""

import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from sanitize_filenames import sanitize_name

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_FILE = "../data_craw/pronouncing_dictionary.txt"
OUTPUT_DIR = "../data_craw/audio_files_new"
CSV_FILE = "../data_craw/audio_data_new.csv"

MAX_WORKERS = 20        # Match crawl_data.py — I/O-bound task, safe to parallelise
BATCH_SIZE = 100
REQUEST_TIMEOUT = 10    # Match crawl_data.py timeout
RETRY_ATTEMPTS = 2      # Reduced — no long blocking delays like before
INTER_BATCH_SLEEP = 1   # Match crawl_data.py inter-batch pause

DICT_API_BASE = "https://api.dictionaryapi.dev/api/v2/entries/en"

CSV_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()

# Thread-local requests.Session — each thread reuses its own TCP connection pool
# instead of opening a new connection for every request (same pattern as
# crawl_data.py but more explicit about connection reuse).
_thread_local = threading.local()

# A valid dictionary word: only ASCII letters, internal hyphens, internal
# apostrophes — no digits, underscores, dots, or symbols.
# Examples accepted : hello, don't, self-aware, a
# Examples rejected : a42128, a.s, aaron_s, !, 'round (leading apostrophe)
_VALID_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z'\-]*[a-zA-Z]$|^[a-zA-Z]$")


# ── Session management ────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    """Return (or create) a per-thread requests.Session with a pooled adapter."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
            max_retries=0,  # retries handled manually
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session


# ── Utilities ─────────────────────────────────────────────────────────────────

def ensure_dir(directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_existing_words() -> set[str]:
    """Return the set of words already present in the CSV (for resume support)."""
    existing: set[str] = set()
    if not os.path.exists(CSV_FILE):
        return existing
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if row:
                existing.add(row[0])
    return existing


def _append_csv(word: str, phoneme: str, audio_path: str) -> None:
    with CSV_LOCK:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([word, phoneme, audio_path])


# ── Word validation ────────────────────────────────────────────────────────────

def is_valid_word(word: str) -> bool:
    """
    Accept only tokens that are genuine English words suitable for dictionary
    lookup. Rejects:
      - CMU pronunciation variant suffixes like WORD(2), WORD(3)
      - Tokens with digits (e.g. a42128)
      - Possessive CMU tokens with underscores (e.g. aaron_s)
      - Abbreviations with dots (e.g. a.s, u.s.a)
      - Pure punctuation / symbols
      - Empty strings
    """
    if not word:
        return False
    w = word.strip()
    if not w:
        return False
    if re.search(r"\(\d+\)$", w):
        return False
    return bool(_VALID_WORD_RE.match(w))


# ── Free Dictionary API ────────────────────────────────────────────────────────

def fetch_audio_url(word: str) -> str | None:
    """
    Query the Free Dictionary API for *word* and return the best audio URL.

    Priority order: US pronunciation > UK pronunciation > any pronunciation.
    Returns None if the word is not found (404) or has no audio entry.
    """
    session = _get_session()
    url = f"{DICT_API_BASE}/{word.lower()}"

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 404:
                return None

            if resp.status_code != 200:
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(0.5)
                continue

            data = resp.json()
            if not isinstance(data, list) or not data:
                return None

            phonetics = data[0].get("phonetics", [])
            us_url: str | None = None
            uk_url: str | None = None
            any_url: str | None = None

            for entry in phonetics:
                audio = entry.get("audio", "").strip()
                if not audio:
                    continue
                al = audio.lower()
                if any_url is None:
                    any_url = audio
                if "-us" in al and us_url is None:
                    us_url = audio
                elif "-uk" in al and uk_url is None:
                    uk_url = audio

            return us_url or uk_url or any_url

        except requests.exceptions.RequestException:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(0.5)

    return None


def download_audio(audio_url: str, output_path: str) -> bool:
    """
    Stream-download an audio file from *audio_url* to *output_path*.
    Cleans up empty or partial files on failure.
    """
    session = _get_session()

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = session.get(audio_url, timeout=REQUEST_TIMEOUT, stream=True)
            if resp.status_code != 200:
                return False

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(output_path) == 0:
                os.remove(output_path)
                return False

            return True

        except Exception:
            if os.path.exists(output_path):
                os.remove(output_path)
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(0.5)

    return False


# ── Per-word processing ────────────────────────────────────────────────────────

def process_word(
    word_data: tuple[str, str],
    existing_words: set[str],
    progress_bar: tqdm,
) -> bool:
    """Download and record the pronunciation audio for a single word."""
    word, phoneme = word_data
    safe_word = sanitize_name(word)

    # Resume: already recorded
    if safe_word in existing_words:
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return True

    audio_filename = f"{safe_word}.mp3"
    audio_path = os.path.join(OUTPUT_DIR, audio_filename)
    relative_path = os.path.join("audio_files_new", audio_filename)

    # Resume: file exists but CSV entry missing
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        _append_csv(safe_word, phoneme, relative_path)
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return True

    # Fetch audio URL from API
    audio_url = fetch_audio_url(word)
    if not audio_url:
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return False

    # Download audio file
    success = download_audio(audio_url, audio_path)
    if success:
        _append_csv(safe_word, phoneme, relative_path)

    with PROGRESS_LOCK:
        progress_bar.update(1)
    return success


# ── Batch processing ───────────────────────────────────────────────────────────

def process_batch(
    batch: list[tuple[str, str]],
    existing_words: set[str],
    progress_bar: tqdm,
) -> None:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_word, wd, existing_words, progress_bar): wd
            for wd in batch
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                wd = futures[future]
                print(f"\nError processing '{wd[0]}': {exc}")


# ── Input parsing ──────────────────────────────────────────────────────────────

def load_word_list(input_file: str) -> list[tuple[str, str]]:
    """
    Parse a CMU-format dictionary file (WORD\\tPHONEME ...).
    Only keeps base forms and words that pass is_valid_word() validation.
    """
    word_list: list[tuple[str, str]] = []
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue

            word = line[:tab_pos]

            if re.search(r"\(\d+\)$", word):
                continue

            phoneme = line[tab_pos + 1:].strip()

            if not is_valid_word(word):
                skipped += 1
                continue

            word_list.append((word, phoneme))

    print(f"Loaded  {len(word_list):>8,} valid words")
    print(f"Skipped {skipped:>8,} invalid / non-dictionary tokens")
    return word_list


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    existing_words = get_existing_words()
    print(f"Already recorded: {len(existing_words):,} words (resume mode)")

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["word", "phoneme", "audio_path"])

    word_list = load_word_list(INPUT_FILE)

    pending = [
        (w, p) for w, p in word_list if sanitize_name(w) not in existing_words
    ]
    print(f"Pending download: {len(pending):,} words\n")

    with tqdm(total=len(pending), desc="Processing words", unit="word") as progress_bar:
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i: i + BATCH_SIZE]
            process_batch(batch, existing_words, progress_bar)
            time.sleep(INTER_BATCH_SLEEP)


if __name__ == "__main__":
    main()
