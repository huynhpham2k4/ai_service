import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sympy import true

from app.core.config import get_settings
from app.schemas.recognize import ErrorResponse, RecognizeResponse
from app.services.pronunciation_service import pronunciation_service

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/mp4",
    "application/octet-stream",  # một số client gửi binary không kèm MIME cụ thể
}

MAX_FILE_SIZE_BYTES = int(settings.MAX_FILE_SIZE_MB * 1024 * 1024)


@router.post(
    "/recognize",
    response_model=RecognizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Đánh giá phát âm",
    description=(
        "Nhận file âm thanh và từ / câu tham chiếu, "
        "trả về điểm phát âm cùng nhận xét chi tiết sử dụng mô hình Wav2Vec2."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "File âm thanh không hợp lệ"},
        413: {"model": ErrorResponse, "description": "File vượt quá kích thước cho phép"},
        422: {"description": "Dữ liệu đầu vào không hợp lệ"},
        500: {"model": ErrorResponse, "description": "Lỗi xử lý nội bộ"},
    },
)
async def recognize(
    audio: Annotated[
        UploadFile,
        File(description="File âm thanh (WAV, MP3, OGG, FLAC, WEBM, ...)"),
    ],
    text: Annotated[
        str,
        Form(
            min_length=1,
            max_length=500,
            description="Từ hoặc câu mà người dùng đang phát âm",
        ),
    ],
    include_phoneme_details: Annotated[
        bool,
        Form(description="Có trả về chi tiết từng âm vị không (mặc định: false)"),
    ] = True,
) -> RecognizeResponse:
    """
    **Đánh giá phát âm với Wav2Vec2**

    - **audio**: File âm thanh chứa giọng đọc của người dùng.
    - **text**: Từ / câu tham chiếu mà người dùng đang cố đọc.
    - **include_phoneme_details**: Nếu `true`, kết quả sẽ kèm phân tích từng ký tự.

    Kết quả trả về gồm:
    - `transcription`: Văn bản nhận dạng từ giọng nói.
    - `score`: Điểm 0–100.
    - `level`: `excellent` / `good` / `fair` / `poor`.
    - `cer` / `wer`: Character / Word Error Rate.
    - `feedback`: Nhận xét bằng tiếng Việt.
    """
    # Kiểm tra kích thước file
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File vượt quá {settings.MAX_FILE_SIZE_MB} MB.",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File âm thanh rỗng.",
        )

    # Kiểm tra MIME type (nếu client gửi kèm)
    content_type = audio.content_type or ""
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Định dạng file không được hỗ trợ: {content_type}.",
        )

    reference_text = text.strip()
    if not reference_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tham số 'text' không được để trống.",
        )

    try:
        result = pronunciation_service.assess(
            audio_bytes=audio_bytes,
            reference_text=reference_text,
            include_phoneme_details=include_phoneme_details,
        )
    except Exception as exc:
        logger.exception("Lỗi khi đánh giá phát âm: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể xử lý file âm thanh. Vui lòng thử lại.",
        ) from exc

    return result
