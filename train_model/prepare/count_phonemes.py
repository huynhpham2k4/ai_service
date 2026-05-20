import csv
from collections import Counter
from pathlib import Path

INPUT_CSV = "../data_craw/audio_data_lower.csv"
OUTPUT_FILE = "../data_craw/phoneme_stats.txt"


def count_phonemes(input_file: str) -> Counter:
    """Đếm tần suất từng phoneme trong cột phoneme."""
    counter = Counter()

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phoneme_str = row.get("phoneme", "").strip()
            if not phoneme_str:
                continue
            for phoneme in phoneme_str.split():
                counter[phoneme] += 1

    return counter


def main():
    input_path = Path(__file__).parent / INPUT_CSV
    output_path = Path(__file__).parent / OUTPUT_FILE

    counter = count_phonemes(input_path)
    unique_phonemes = sorted(counter.keys())

    print(f"Total unique phonemes: {len(unique_phonemes)}\n")
    print("Phoneme list (alphabetical):")
    for i, p in enumerate(unique_phonemes, 1):
        print(f"  {i:2}. {p:6}  (count: {counter[p]:,})")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Total unique phonemes: {len(unique_phonemes)}\n\n")
        f.write("phoneme\tcount\n")
        for p in unique_phonemes:
            f.write(f"{p}\t{counter[p]}\n")

    print(f"\nSaved details to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
