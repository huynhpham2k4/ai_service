import os
import csv
import requests
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from tqdm import tqdm

# Constants
DICTIONARY_URL = 'https://dict.laban.vn/ajax/getsound'
INPUT_FILE = 'data_draw/pronouncing_dictionary.txt'
OUTPUT_DIR = 'data_draw/audio_files'
CSV_FILE = 'data_draw/audio_data.csv'
MAX_WORKERS = 20  # Tăng số lượng thread
BATCH_SIZE = 100  # Số lượng từ xử lý mỗi batch
CSV_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_existing_words():
    """Get set of words that already exist in CSV"""
    existing_words = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Check if row is not empty
                    existing_words.add(row[0])
    return existing_words

def download_audio(word, accent, output_path):
    """Download audio file for a word with given accent"""
    try:
        # Remove any quotes from the word
        clean_word = word.strip("'")
        
        # First request to get the audio URL
        response = requests.get(DICTIONARY_URL, params={
            'word': clean_word,
            'accent': accent
        }, timeout=10)  # Add timeout
        
        if response.status_code == 200:
            try:
                # Parse JSON response
                json_response = response.json()
                
                # Check if there's an error
                if json_response.get('error', 1) != 0:
                    return False
                
                # Get audio URL from response
                audio_url = json_response.get('data')
                if not audio_url:
                    return False
                
                # Download the actual audio file
                audio_response = requests.get(audio_url, timeout=10)  # Add timeout
                if audio_response.status_code == 200:
                    # Save file with proper handling
                    try:
                        with open(output_path, 'wb') as f:
                            f.write(audio_response.content)
                        # Verify file was written
                        if os.path.getsize(output_path) == 0:
                            return False
                        return True
                    except Exception:
                        return False
                return False
                    
            except json.JSONDecodeError:
                return False
        return False
    except Exception:
        return False

def process_word(word_data, existing_words, progress_bar):
    """Process a single word with its phoneme"""
    word, phoneme = word_data
    
    # Skip if word already exists
    if word in existing_words:
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return True
    
    # Create audio filename
    audio_filename = f"{word}.mp3"
    audio_path = os.path.join(OUTPUT_DIR, audio_filename)
    relative_path = os.path.join('audio_files', audio_filename)
    
    # Skip if audio file already exists
    if os.path.exists(audio_path):
        # Add to CSV if not already there
        with CSV_LOCK:
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([word, phoneme, relative_path])
        with PROGRESS_LOCK:
            progress_bar.update(1)
        return True
    
    # Download audio file
    success = download_audio(word, 'us', audio_path)
    
    if success:
        # Write to CSV with thread safety
        with CSV_LOCK:
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([word, phoneme, relative_path])
    
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
    # Create necessary directories
    ensure_dir(OUTPUT_DIR)
    
    # Get existing words from CSV
    existing_words = get_existing_words()
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['word', 'phoneme', 'audio_path'])
    
    # Read all words from dictionary file
    word_list = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            first_space = line.find(' ')
            if first_space == -1:
                continue
            
            word = line[:first_space]
            phoneme = line[first_space:].strip()
            word_list.append((word, phoneme))
    
    # Create progress bar
    with tqdm(total=len(word_list), desc="Processing words") as progress_bar:
        # Process words in batches
        for i in range(0, len(word_list), BATCH_SIZE):
            batch = word_list[i:i + BATCH_SIZE]
            process_batch(batch, existing_words, progress_bar)
            time.sleep(1)  # Small delay between batches

if __name__ == "__main__":
    main() 