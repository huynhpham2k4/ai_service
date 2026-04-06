from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class PronunciationLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class PhonemeDetail(BaseModel):
    expected: str = Field(..., description="Âm vị kỳ vọng")
    recognized: str = Field(..., description="Âm vị nhận dạng được")
    is_correct: bool = Field(..., description="Có phát âm đúng không")


class RecognizeResponse(BaseModel):
    transcription: str = Field(..., description="Văn bản nhận dạng từ âm thanh")
    reference_text: str = Field(..., description="Văn bản tham chiếu (từ người dùng nhập)")
    score: float = Field(..., ge=0.0, le=100.0, description="Điểm phát âm (0-100)")
    level: PronunciationLevel = Field(..., description="Mức độ phát âm")
    cer: float = Field(..., ge=0.0, description="Character Error Rate")
    wer: float = Field(..., ge=0.0, description="Word Error Rate")
    feedback: str = Field(..., description="Nhận xét về phát âm")
    phoneme_details: Optional[list[PhonemeDetail]] = Field(
        default=None, description="Chi tiết từng âm vị"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "hello world",
                "reference_text": "hello world",
                "score": 95.0,
                "level": "excellent",
                "cer": 0.0,
                "wer": 0.0,
                "feedback": "Phát âm rất tốt! Tiếp tục phát huy.",
                "phoneme_details": None,
            }
        }


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Mô tả lỗi")
    code: str = Field(..., description="Mã lỗi")
