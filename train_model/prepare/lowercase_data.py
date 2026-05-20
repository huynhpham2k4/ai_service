import csv
import os
from pathlib import Path

INPUT_CSV = "../data_craw/audio_data.csv"
OUTPUT_CSV = "../data_craw/audio_data_lower.csv"
AUDIO_DIR = "../data_craw/audio_files"


def rename_audio_files(audio_dir: str) -> dict[str, str]:
    """Đổi tên tất cả file MP3 sang chữ thường, trả về mapping tên cũ → tên mới."""
    renamed = {}
    audio_path = Path(audio_dir)

    for file in audio_path.glob("*.mp3"):
        lower_name = file.name.lower()
        lower_path = audio_path / lower_name

        if file.name == lower_name:
            continue

        if lower_path.exists():
            print(f"[SKIP] Đã tồn tại: {lower_name}")
        else:
            file.rename(lower_path)
            print(f"[RENAME] {file.name} → {lower_name}")

        renamed[file.name] = lower_name

    return renamed


def lowercase_csv(input_file: str, output_file: str) -> None:
    """Chuyển toàn bộ word, phoneme, audio_path sang chữ thường."""
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "word": row["word"].lower(),
                "phoneme": row["phoneme"].lower(),
                "audio_path": row["audio_path"].lower(),
            })

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "phoneme", "audio_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Đã lưu CSV chữ thường vào: {output_file}")
    print(f"       Tổng số dòng: {len(rows)}")


def main():
    print("=== Bước 1: Đổi tên file audio sang chữ thường ===")
    rename_audio_files(AUDIO_DIR)

    print("\n=== Bước 2: Chuyển CSV sang chữ thường ===")
    lowercase_csv(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
