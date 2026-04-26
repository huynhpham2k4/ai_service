import asyncio
import logging
import sys
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.service import predict_phonemes, expected_phonemes, compute_score, phonemes_to_ipa

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()

MAX_FILE_SIZE = int(settings.MAX_FILE_SIZE_MB * 1024 * 1024)

app = FastAPI(title="Pronunciation Assessment API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(
    audio: Annotated[UploadFile, File(description="File âm thanh")],
    text: Annotated[str, Form(min_length=1, max_length=500, description="Từ hoặc câu cần phát âm")],
):
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="File âm thanh rỗng.")
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File quá lớn.")

    text = text.strip()
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Text không được để trống.")

    try:
        loop = asyncio.get_event_loop()
        predicted = await loop.run_in_executor(None, predict_phonemes, audio_bytes)
        expected = await loop.run_in_executor(None, expected_phonemes, text)
        score = compute_score(expected, predicted)

        return {
            "reference_text": text,
            "expected_phonemes": phonemes_to_ipa(expected),
            "predicted_phonemes": phonemes_to_ipa(predicted),
            "score": score,
        }
    except Exception as exc:
        logger.exception("Lỗi xử lý: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Không thể xử lý file âm thanh.") from exc
