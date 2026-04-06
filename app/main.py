import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.service import ErrorResponse, RecognizeResponse, pronunciation_service

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac", "audio/webm", "audio/mp4",
    "application/octet-stream",
}
MAX_FILE_SIZE_BYTES = int(settings.MAX_FILE_SIZE_MB * 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Khởi động %s v%s ...", settings.APP_NAME, settings.APP_VERSION)
    yield
    logger.info("Tắt ứng dụng.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API đánh giá phát âm tiếng Anh sử dụng mô hình Wav2Vec2.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"], summary="Kiểm tra trạng thái API")
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.post(
    "/recognize",
    response_model=RecognizeResponse,
    tags=["Pronunciation Assessment"],
    summary="Đánh giá phát âm",
    responses={
        400: {"model": ErrorResponse, "description": "File âm thanh không hợp lệ"},
        413: {"model": ErrorResponse, "description": "File vượt quá kích thước cho phép"},
        500: {"model": ErrorResponse, "description": "Lỗi xử lý nội bộ"},
    },
)
async def recognize(
    audio: Annotated[UploadFile, File(description="File âm thanh (WAV, MP3, OGG, FLAC, ...)")],
    text: Annotated[str, Form(min_length=1, max_length=500, description="Từ hoặc câu cần phát âm")],
    include_phoneme_details: Annotated[bool, Form(description="Trả về chi tiết âm vị")] = True,
) -> RecognizeResponse:
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="File âm thanh rỗng.")
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File vượt quá {settings.MAX_FILE_SIZE_MB} MB.")

    content_type = audio.content_type or ""
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Định dạng không được hỗ trợ: {content_type}.")

    reference_text = text.strip()
    if not reference_text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Tham số 'text' không được để trống.")

    try:
        return pronunciation_service.assess(audio_bytes, reference_text, include_phoneme_details)
    except Exception as exc:
        logger.exception("Lỗi đánh giá phát âm: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Không thể xử lý file âm thanh.") from exc
