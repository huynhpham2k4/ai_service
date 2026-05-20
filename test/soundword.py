import asyncio
from datetime import datetime
from pathlib import Path

import edge_tts

# Giọng nam mặc định theo mã ngôn ngữ (Edge neural)
_LANG_TO_VOICE: dict[str, str] = {
    "en": "en-US-GuyNeural",
    "en-us": "en-US-GuyNeural",
    "en-gb": "en-GB-RyanNeural",
    "vi": "vi-VN-NamMinhNeural",
}


def _resolve_voice(lang: str, voice: str | None) -> str:
    if voice:
        return voice.strip()
    key = lang.strip().lower().replace("_", "-")
    if key in _LANG_TO_VOICE:
        return _LANG_TO_VOICE[key]
    if key.startswith("en"):
        return _LANG_TO_VOICE["en"]
    return _LANG_TO_VOICE["en"]


def create_audio_from_word(
    word: str,
    output_dir: str = "test/audio_samples",
    filename: str | None = None,
    lang: str = "en",
    slow: bool = False,
    voice: str | None = None,
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
    voice_id = _resolve_voice(lang, voice)
    rate = "-30%" if slow else "+0%"

    async def _save() -> None:
        communicate = edge_tts.Communicate(clean_word, voice_id, rate=rate)
        await communicate.save(str(output_path))

    asyncio.run(_save())
    return str(output_path.resolve())


if __name__ == "__main__":
    audio_path = create_audio_from_word("dictionary", lang="en")
    print(f"Da tao file: {audio_path}")
