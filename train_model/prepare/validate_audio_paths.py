import csv
import re
from pathlib import Path

csv_path = Path(__file__).parent / "../data_craw/audio_data_lower.csv"
audio_dir = Path(__file__).parent / "../data_craw/audio_files"
unsafe = re.compile(r"[^a-z0-9._-]")

bad_csv = missing = 0
with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        stem = Path(row["audio_path"]).stem
        if "'" in row["word"] or "'" in row["audio_path"] or unsafe.search(stem):
            bad_csv += 1
        if not (audio_dir / f"{stem}.mp3").exists():
            missing += 1

apost = list(audio_dir.glob("*'*.mp3"))
print("csv unsafe rows:", bad_csv)
print("missing audio files:", missing)
print("files with apostrophe:", len(apost))
