import asyncio
import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import edge_tts
from tqdm import tqdm

from sanitize_filenames import sanitize_name

# Constants
TTS_VOICE = "en-US-GuyNeural"
INPUT_FILE = "../data_craw/pronouncing_dictionary.txt"
OUTPUT_DIR = "../data_craw/audio_files"
CSV_FILE = "../data_craw/audio_data.csv"
MAX_WORKERS = 10
BATCH_SIZE = 100
CSV_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()


def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_existing_words():
    """Get set of words that already exist in CSV"""
    existing_words = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    existing_words.add(row[0])
    return existing_words


def prepare_word_for_tts(word: str) -> str | None:
    """Chuẩn hóa từ CMU thành text có thể đọc bằng TTS."""
    clean = word.strip("'")
    clean = clean.replace(".", " ").strip()
    if not clean or not re.search(r"[a-zA-Z]", clean):
        return None
    return clean


async def _save_tts(text: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(output_path)


def generate_audio(word: str, output_path: str) -> bool:
    """Tạo file audio bằng edge-tts."""
    text = prepare_word_for_tts(word)
    if not text:
        return False

    try:
        asyncio.run(_save_tts(text, output_path))
        if os.path.getsize(output_path) == 0:
            return False
        return True
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def process_word(word_data, existing_words, progress_bar):
    """Process a single word with its phoneme"""
    word, phoneme = word_data

    if word in existing_words:
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return True

    safe_word = sanitize_name(word)
    audio_filename = f"{safe_word}.mp3"
    audio_path = os.path.join(OUTPUT_DIR, audio_filename)
    relative_path = os.path.join("audio_files", audio_filename)

    if os.path.exists(audio_path):
        with CSV_LOCK:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([safe_word, phoneme, relative_path])
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return True

    success = generate_audio(word, audio_path)

    if success:
        with CSV_LOCK:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([safe_word, phoneme, relative_path])

    with PROGRESS_LOCK:
        progress_bar.update(1)
    return success


def process_batch(batch, existing_words, progress_bar):
    """Process a batch of words using thread pool"""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_word, word_data, existing_words, progress_bar): word_data
            for word_data in batch
        }

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                word_data = futures[future]
                print(f"\nError processing {word_data[0]}: {str(e)}")


def main():
    ensure_dir(OUTPUT_DIR)

    existing_words = get_existing_words()

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["word", "phoneme", "audio_path"])

    word_list = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue

            word = line[:tab_pos]
            # Bỏ qua các biến thể phát âm (2), (3)... chỉ lấy từ gốc
            if re.search(r"\(\d+\)$", word):
                continue
            phoneme = line[tab_pos + 1 :].strip()
            word_list.append((word, phoneme))

    with tqdm(total=len(word_list), desc="Processing words") as progress_bar:
        for i in range(0, len(word_list), BATCH_SIZE):
            batch = word_list[i : i + BATCH_SIZE]
            process_batch(batch, existing_words, progress_bar)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
