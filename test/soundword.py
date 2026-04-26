from pathlib import Path
from datetime import datetime

from gtts import gTTS


def create_audio_from_word(
    word: str,
    output_dir: str = "test/audio_samples",
    filename: str | None = None,
    lang: str = "en",
    slow: bool = False,
) -> str:
    clean_word = word.strip()
    if not clean_word:
        raise ValueError("word khong duoc de trong")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        safe_text = "".join(ch for ch in clean_word if ch.isalnum() or ch in ("-", "_", " "))
        safe_text = "_".join(safe_text.split())[:40] or "audio"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_text}_{timestamp}.mp3"
    elif not filename.lower().endswith(".mp3"):
        filename = f"{filename}.mp3"

    output_path = out_dir / filename
    gTTS(text=clean_word, lang=lang, slow=slow).save(str(output_path))
    return str(output_path.resolve())


if __name__ == "__main__":
    audio_path = create_audio_from_word("public", lang="en")
    print(f"Da tao file: {audio_path}")
