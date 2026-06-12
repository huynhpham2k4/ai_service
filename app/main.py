import asyncio
import logging
import sys
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.service import (
    predict_phonemes,
    expected_phonemes,
    compute_score,
    phonemes_to_ipa,
    phonemes_to_ipa_tokens,
    align_and_trim_noise,
    _load_model,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()

MAX_FILE_SIZE = int(settings.MAX_FILE_SIZE_MB * 1024 * 1024)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    if settings.DEBUG_SAVE_AUDIO:
        logger.info(
            "DEBUG_SAVE_AUDIO=bật — WAV lưu tại %s (DEBUG_PLAY_AUDIO=%s)",
            settings.DEBUG_AUDIO_DIR,
            settings.DEBUG_PLAY_AUDIO,
        )
    yield


app = FastAPI(title="Pronunciation Assessment API", lifespan=lifespan)
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
        expected_ipa = phonemes_to_ipa_tokens(expected)
        predicted_ipa = phonemes_to_ipa_tokens(predicted)
        
        alignment = align_and_trim_noise(expected_ipa, predicted_ipa)
        
        aligned_expected_str = " ".join(alignment["aligned_expected"])
        aligned_predicted_str = " ".join(alignment["aligned_predicted"])
        score = alignment["score"]

        logger.info("Reference text: %s", text)
        logger.info("Expected phonemes (raw): %s", expected)
        logger.info("Predicted phonemes (raw): %s", predicted)
        logger.info("Aligned expected: %s", aligned_expected_str)
        logger.info("Aligned predicted: %s", aligned_predicted_str)
        logger.info("Distance: %d", alignment["distance"])
        logger.info("Score: %.2f", score)

        return {
            "reference_text": text,
            "expected_phonemes": aligned_expected_str,
            "predicted_phonemes": aligned_predicted_str,
            "score": score,
        }   
    except Exception as exc:
        logger.exception("Lỗi xử lý: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Không thể xử lý file âm thanh.") from exc
