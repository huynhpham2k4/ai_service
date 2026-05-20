"""
Doi ten file audio va cap nhat CSV: chi dung a-z, 0-9, _, -, .
Apostrophe (') va ky tu dac biet khac -> _
"""
import csv
import re
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "../data_craw/audio_files"
CSV_FILES = [
    Path(__file__).parent / "../data_craw/audio_data_lower.csv",
    Path(__file__).parent / "../data_craw/audio_data.csv",
    Path(__file__).parent / "../data_craw/audio_data_template.csv",
]
SAFE_PATTERN = re.compile(r"[^a-z0-9._-]")


def sanitize_name(name: str) -> str:
    """Chuyen ten thanh ky tu an toan cho Kaggle (a-z, 0-9, _, -, .)."""
    s = name.lower()
    # Possessive cuoi tu: accountants' -> accountants_s (tranh trung accountants.mp3)
    if s.endswith("'"):
        s = s[:-1] + "_s"
    else:
        s = s.replace("'", "_")
    s = SAFE_PATTERN.sub("_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"\.+", ".", s)
    return s.strip("._")


def sanitize_audio_path(audio_path: str) -> str:
    """audio_files\\foo's.mp3 -> audio_files\\foo_s.mp3"""
    parts = audio_path.replace("/", "\\").split("\\")
    if len(parts) >= 2:
        parts[-1] = sanitize_name(Path(parts[-1]).stem) + ".mp3"
    else:
        parts[0] = sanitize_name(Path(parts[0]).stem) + ".mp3"
    return "\\".join(parts)


def build_rename_map(audio_dir: Path) -> dict[str, str]:
    """old_stem -> new_stem (chi khi khac nhau)."""
    mapping = {}
    used = set()

    for f in sorted(audio_dir.glob("*.mp3")):
        old_stem = f.stem
        new_stem = sanitize_name(old_stem)
        if old_stem == new_stem:
            used.add(new_stem)
            continue

        candidate = new_stem
        n = 2
        while candidate in used and candidate != old_stem:
            candidate = f"{new_stem}_{n}"
            n += 1

        if candidate != old_stem:
            mapping[old_stem] = candidate
            used.add(candidate)
        else:
            used.add(old_stem)

    return mapping


def rename_audio_files(audio_dir: Path, mapping: dict[str, str]) -> int:
    count = 0
    for old_stem, new_stem in mapping.items():
        old_path = audio_dir / f"{old_stem}.mp3"
        if not old_path.exists():
            continue

        candidate = new_stem
        n = 2
        while (audio_dir / f"{candidate}.mp3").exists() and (
            audio_dir / f"{candidate}.mp3"
        ).resolve() != old_path.resolve():
            candidate = f"{new_stem}_{n}"
            n += 1

        new_path = audio_dir / f"{candidate}.mp3"
        if old_path.name != new_path.name:
            old_path.rename(new_path)
            mapping[old_stem] = candidate
            count += 1
    return count


def update_csv(csv_path: Path, mapping: dict[str, str]) -> int:
    if not csv_path.exists():
        return 0

    rows = []
    updated = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            word = row["word"]
            new_word = mapping.get(word, sanitize_name(word))
            if new_word != word:
                updated += 1
            row["word"] = new_word

            old_path = row["audio_path"]
            stem = Path(old_path).stem
            new_stem = mapping.get(stem, sanitize_name(stem))
            row["audio_path"] = f"audio_files\\{new_stem}.mp3"
            rows.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def main():
    audio_dir = AUDIO_DIR.resolve()
    mapping = build_rename_map(audio_dir)

    print(f"Files can rename: {len(mapping)}")
    if mapping:
        samples = list(mapping.items())[:5]
        for old, new in samples:
            print(f"  {old}.mp3 -> {new}.mp3")

    renamed = rename_audio_files(audio_dir, mapping)
    print(f"\nRenamed audio files: {renamed}")

    for csv_path in CSV_FILES:
        if csv_path.exists():
            n = update_csv(csv_path, mapping)
            print(f"Updated rows in {csv_path.name}: {n}")

    remaining = sum(
        1 for f in audio_dir.glob("*.mp3") if SAFE_PATTERN.search(f.stem.replace("_", "X")) or "'" in f.name
    )
    # recheck apostrophe and unsafe
    bad = [f.name for f in audio_dir.glob("*.mp3") if "'" in f.name or re.search(r"[^a-z0-9._-]", f.stem)]
    print(f"\nRemaining unsafe filenames: {len(bad)}")
    if bad[:5]:
        print("  Examples:", bad[:5])


if __name__ == "__main__":
    main()
