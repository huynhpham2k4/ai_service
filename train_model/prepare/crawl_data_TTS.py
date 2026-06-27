from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import edge_tts

try:
    from wordfreq import zipf_frequency
except Exception:  # pragma: no cover
    zipf_frequency = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from sanitize_filenames import sanitize_name

logger = logging.getLogger("cmudict_tts")


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning / filtering rules
# ──────────────────────────────────────────────────────────────────────────────

_CMU_VARIANT_RE = re.compile(r"^([A-Z]+)\((\d+)\)$")
_ALPHA_ONLY_RE = re.compile(r"^[A-Z]+$")
_CSV_HEADER = ("word", "phoneme", "audio_path")


@dataclass(frozen=True)
class Config:
    # IO
    # Defaults are resolved relative to this file (not current working directory)
    input_file: str = ""
    output_dir: str = ""
    csv_file: str = ""

    # edge-tts
    voice: str = "en-US-GuyNeural"
    audio_ext: str = "mp3"

    # Async concurrency
    concurrency: int = 20
    queue_maxsize: int = 2000

    # Retry
    max_attempts: int = 4
    base_backoff_s: float = 0.7
    max_backoff_s: float = 8.0

    # Common vocabulary filter (wordfreq)
    # Zipf scale guideline: 6=very common, 5=common, 4=moderate, 3=rare-ish.
    min_zipf: float = 4.0

    # Additional heuristics to keep dataset “learnable”
    min_len: int = 2
    max_len: int = 20

    # Logging
    log_every: int = 200
    print_each_success: bool = False
    print_each_fail: bool = False
    use_tqdm: bool = True


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _default_paths() -> tuple[str, str, str]:
    """
    Resolve default IO paths relative to this script:
      ai_service/train_model/prepare/crawl_data_TTS.py
        -> ai_service/train_model/data_craw/...
    """
    base = Path(__file__).resolve().parent  # .../train_model/prepare
    data_craw = (base / ".." / "data_craw").resolve()
    input_file = str(data_craw / "pronouncing_dictionary.txt")
    output_dir = str(data_craw / "audio_files_new")
    csv_file = str(data_craw / "audio_data_new.csv")
    return input_file, output_dir, csv_file


def _is_header_row(row: list[str]) -> bool:
    if len(row) < 3:
        return False
    return row[0].strip().lower() == "word" and row[1].strip().lower() == "phoneme"


def _csv_init_if_needed(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_CSV_HEADER)


def ensure_csv_header(csv_path: str) -> None:
    """Add header if missing; deduplicate rows by word (keeps first occurrence)."""
    if not os.path.exists(csv_path):
        _csv_init_if_needed(csv_path)
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_HEADER)
        return

    if _is_header_row(rows[0]):
        return

    logger.info("CSV has no header — adding header and removing duplicate words...")
    unique: dict[str, list[str]] = {}
    for row in rows:
        if not row or len(row) < 3:
            continue
        word = row[0].strip()
        if word and word not in unique:
            unique[word] = [row[0], row[1], row[2]]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        writer.writerows(unique.values())

    logger.info("CSV fixed: %d unique rows (+ header)", len(unique))


def get_existing_words(csv_path: str) -> set[str]:
    existing: set[str] = set()
    if not os.path.exists(csv_path):
        return existing
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            if i == 0 and _is_header_row(row):
                continue
            existing.add(row[0].strip())
    return existing


async def _append_csv_row(
    csv_lock: asyncio.Lock,
    csv_words: set[str],
    cfg: Config,
    word: str,
    phoneme: str,
    rel_audio: str,
) -> bool:
    """Append one row only if word is not already recorded in CSV."""
    if word in csv_words:
        return False
    async with csv_lock:
        if word in csv_words:
            return False
        with open(cfg.csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([word, phoneme, rel_audio])
        csv_words.add(word)
    return True


def iter_cmudict_entries(path: str) -> Iterable[tuple[str, str]]:
    """
    Streams CMUdict-like file lines: WORD<TAB>PHONEME...
    Skips empty/malformed rows.
    """
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            tab = line.find("\t")
            if tab == -1:
                continue
            word = line[:tab].strip()
            phoneme = line[tab + 1 :].strip()
            if not word or not phoneme:
                continue
            yield word, phoneme


def normalize_cmu_word(token: str) -> str | None:
    """
    CMUdict often contains:
      - pronunciation variants WORD(2) -> need to discard entirely (no mapping)
      - symbols / punctuation / special tokens -> remove
    Keep only A-Z, then lower().
    """
    t = token.strip()
    if not t:
        return None
    if _CMU_VARIANT_RE.match(t):
        return None
    if not _ALPHA_ONLY_RE.match(t):
        return None
    return t.lower()


def is_common_english_word(word: str, cfg: Config) -> bool:
    """
    Frequency filter:
      - if wordfreq is installed -> zipf_frequency(word, 'en') >= cfg.min_zipf
      - otherwise fallback: let it pass (so script works), but log warning
    """
    if zipf_frequency is None:
        logger.warning(
            "Library 'wordfreq' is not installed — frequency filter is disabled. "
            "Recommended: pip install wordfreq"
        )
        return True
    try:
        return zipf_frequency(word, "en") >= cfg.min_zipf
    except Exception:
        return False


def is_good_vocab_candidate(word: str, cfg: Config) -> bool:
    if not (cfg.min_len <= len(word) <= cfg.max_len):
        return False
    # minimal protection against "words" like 'aaaaa'
    if len(set(word)) == 1:
        return False
    return is_common_english_word(word, cfg)


def audio_paths(word: str, cfg: Config) -> tuple[str, str]:
    safe = sanitize_name(word)
    filename = f"{safe}.{cfg.audio_ext}"
    abs_path = os.path.join(cfg.output_dir, filename)
    rel_path = os.path.join(Path(cfg.output_dir).name, filename)
    return abs_path, rel_path


async def save_tts_with_retry(text: str, out_path: str, cfg: Config) -> bool:
    """
    edge_tts async save with retries and exponential backoff.
    """
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            communicate = edge_tts.Communicate(text, cfg.voice)
            await communicate.save(out_path)
            if os.path.getsize(out_path) == 0:
                raise RuntimeError("empty audio file")
            return True
        except Exception as e:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass

            if attempt >= cfg.max_attempts:
                logger.debug("TTS failed for '%s' (%s)", text, e)
                return False

            backoff = min(cfg.max_backoff_s, cfg.base_backoff_s * (2 ** (attempt - 1)))
            backoff = backoff * (0.8 + 0.4 * random.random())
            await asyncio.sleep(backoff)
    return False


async def worker(
    name: str,
    queue: asyncio.Queue[tuple[str, str]],
    csv_lock: asyncio.Lock,
    csv_words: set[str],
    pbar: "tqdm | None",
    pbar_lock: asyncio.Lock,
    seen_words: set[str],
    cfg: Config,
) -> None:
    processed = 0
    ok_count = 0
    fail_count = 0
    while True:
        item = await queue.get()
        if item is None:  # type: ignore[comparison-overlap]
            queue.task_done()
            return

        word, phoneme = item
        queue.task_done()
        processed += 1

        abs_audio, rel_audio = audio_paths(word, cfg)

        # Skip if audio exists already (fast resume even if CSV lost)
        if os.path.exists(abs_audio) and os.path.getsize(abs_audio) > 0:
            if await _append_csv_row(csv_lock, csv_words, cfg, word, phoneme, rel_audio):
                ok_count += 1
            if cfg.print_each_success:
                logger.info("[%s] SKIP (exists) %s", name, word)
            if pbar is not None:
                async with pbar_lock:
                    pbar.update(1)
            continue

        ok = await save_tts_with_retry(word, abs_audio, cfg)
        if ok:
            if await _append_csv_row(csv_lock, csv_words, cfg, word, phoneme, rel_audio):
                ok_count += 1
            if cfg.print_each_success:
                logger.info("[%s] OK %s -> %s", name, word, rel_audio)
        else:
            fail_count += 1
            if cfg.print_each_fail:
                logger.warning("[%s] FAIL %s", name, word)

        if pbar is not None:
            async with pbar_lock:
                pbar.update(1)

        if processed % cfg.log_every == 0:
            logger.info(
                "[%s] processed=%d ok=%d fail=%d (seen=%d, queue=%d)",
                name,
                processed,
                ok_count,
                fail_count,
                len(seen_words),
                queue.qsize(),
            )


async def run(cfg: Config) -> None:
    ensure_dir(cfg.output_dir)
    _csv_init_if_needed(cfg.csv_file)
    ensure_csv_header(cfg.csv_file)

    csv_words = get_existing_words(cfg.csv_file)
    logger.info("Resume: already in CSV = %d", len(csv_words))

    # Stage 1: stream parse + clean + dedup + frequency filter
    seen: set[str] = set(csv_words)
    accepted: list[tuple[str, str]] = []
    skipped = 0
    duplicates = 0

    for raw_word, phoneme in iter_cmudict_entries(cfg.input_file):
        word = normalize_cmu_word(raw_word)
        if word is None:
            skipped += 1
            continue
        if word in seen:
            duplicates += 1
            continue
        if not is_good_vocab_candidate(word, cfg):
            skipped += 1
            continue
        seen.add(word)
        accepted.append((word, phoneme))

    logger.info("Accepted=%d | Skipped=%d | Duplicates=%d", len(accepted), skipped, duplicates)

    # Stage 2: async edge-tts generation
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=cfg.queue_maxsize)
    csv_lock = asyncio.Lock()
    pbar_lock = asyncio.Lock()

    pbar = None
    if cfg.use_tqdm:
        if tqdm is None:
            logger.warning("tqdm is not installed; progress bar disabled.")
        else:
            pbar = tqdm(total=len(accepted), desc="Crawling TTS", unit="word")

    workers = [
        asyncio.create_task(
            worker(f"w{i+1}", queue, csv_lock, csv_words, pbar, pbar_lock, seen, cfg)
        )
        for i in range(cfg.concurrency)
    ]

    for item in accepted:
        await queue.put(item)

    # stop signals
    for _ in workers:
        await queue.put(None)  # type: ignore[arg-type]

    await queue.join()
    await asyncio.gather(*workers)

    if pbar is not None:
        pbar.close()

    logger.info("Done. CSV: %s | Audio dir: %s", cfg.csv_file, cfg.output_dir)


def parse_args() -> Config:
    default_input, default_out_dir, default_csv = _default_paths()
    p = argparse.ArgumentParser(description="Clean CMUdict + generate edge-tts audio for common English words.")
    p.add_argument("--input", default=default_input, help="Path to CMU dict text file")
    p.add_argument("--out-dir", default=default_out_dir, help="Output directory for audio files")
    p.add_argument("--csv", default=default_csv, help="Output CSV path")
    p.add_argument("--voice", default=Config.voice, help="edge-tts voice")
    p.add_argument("--min-zipf", type=float, default=Config.min_zipf, help="wordfreq zipf threshold (common vocab)")
    p.add_argument("--min-len", type=int, default=Config.min_len, help="Minimum word length to keep")
    p.add_argument("--max-len", type=int, default=Config.max_len, help="Maximum word length to keep")
    p.add_argument("--concurrency", type=int, default=Config.concurrency, help="Number of async workers")
    p.add_argument("--log-every", type=int, default=Config.log_every, help="Log progress every N items per worker")
    p.add_argument("--print-each-success", action="store_true", help="Print every successful word (very verbose)")
    p.add_argument("--print-each-fail", action="store_true", help="Print every failed word (verbose)")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
    args = p.parse_args()
    return Config(
        input_file=args.input,
        output_dir=args.out_dir,
        csv_file=args.csv,
        voice=args.voice,
        min_zipf=args.min_zipf,
        min_len=args.min_len,
        max_len=args.max_len,
        concurrency=args.concurrency,
        log_every=max(1, args.log_every),
        print_each_success=bool(args.print_each_success),
        print_each_fail=bool(args.print_each_fail),
        use_tqdm=not bool(args.no_tqdm),
    )


def main() -> None:
    _setup_logging()
    cfg = parse_args()
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
