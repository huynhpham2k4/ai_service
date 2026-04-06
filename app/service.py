import io
import difflib
import logging
from enum import Enum
from typing import Optional

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PronunciationLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class PhonemeDetail(BaseModel):
    expected: str = Field(..., description="Âm vị kỳ vọng")
    recognized: str = Field(..., description="Âm vị nhận dạng được")
    is_correct: bool = Field(..., description="Phát âm đúng không")


class RecognizeResponse(BaseModel):
    transcription: str = Field(..., description="Văn bản nhận dạng từ âm thanh")
    reference_text: str = Field(..., description="Văn bản tham chiếu")
    score: float = Field(..., ge=0.0, le=100.0, description="Điểm phát âm (0-100)")
    level: PronunciationLevel = Field(..., description="Mức độ phát âm")
    cer: float = Field(..., ge=0.0, description="Character Error Rate")
    wer: float = Field(..., ge=0.0, description="Word Error Rate")
    feedback: str = Field(..., description="Nhận xét về phát âm")
    phoneme_details: Optional[list[PhonemeDetail]] = Field(default=None)

    model_config = {
        "json_schema_extra": {
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
    }


class ErrorResponse(BaseModel):
    detail: str
    code: str


# ── Metrics ───────────────────────────────────────────────────────────────────

def _levenshtein(seq1: list, seq2: list) -> int:
    dp = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1):
        new_dp = [i + 1]
        for j, b in enumerate(seq2):
            new_dp.append(dp[j] if a == b else 1 + min(dp[j], dp[j + 1], new_dp[-1]))
        dp = new_dp
    return dp[len(seq2)]


def _compute_cer(reference: str, hypothesis: str) -> float:
    ref = list(reference.lower().replace(" ", ""))
    hyp = list(hypothesis.lower().replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _compute_wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _compute_score(cer: float, wer: float) -> float:
    return round(max(0.0, min(100.0, (1.0 - 0.6 * wer - 0.4 * cer) * 100.0)), 2)


def _get_level(score: float) -> PronunciationLevel:
    if score >= settings.SCORE_EXCELLENT:
        return PronunciationLevel.EXCELLENT
    if score >= settings.SCORE_GOOD:
        return PronunciationLevel.GOOD
    if score >= settings.SCORE_FAIR:
        return PronunciationLevel.FAIR
    return PronunciationLevel.POOR


def _get_feedback(level: PronunciationLevel, cer: float, wer: float) -> str:
    return {
        PronunciationLevel.EXCELLENT: "Phát âm rất tốt! Tiếp tục phát huy.",
        PronunciationLevel.GOOD: "Phát âm tốt, chỉ có một số lỗi nhỏ.",
        PronunciationLevel.FAIR: f"Phát âm trung bình. CER={cer:.2%}, WER={wer:.2%}. Cần luyện tập thêm.",
        PronunciationLevel.POOR: f"Phát âm cần cải thiện. CER={cer:.2%}, WER={wer:.2%}. Hãy nghe lại và luyện tập.",
    }[level]


def _build_phoneme_details(reference: str, hypothesis: str) -> list[PhonemeDetail]:
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    details: list[PhonemeDetail] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref_chars, hyp_chars).get_opcodes():
        if tag == "equal":
            details += [PhonemeDetail(expected=ref_chars[i1+k], recognized=hyp_chars[j1+k], is_correct=True) for k in range(i2-i1)]
        elif tag == "replace":
            for k in range(max(i2-i1, j2-j1)):
                details.append(PhonemeDetail(
                    expected=ref_chars[i1+k] if k < i2-i1 else "-",
                    recognized=hyp_chars[j1+k] if k < j2-j1 else "-",
                    is_correct=False,
                ))
        elif tag == "delete":
            details += [PhonemeDetail(expected=ref_chars[i1+k], recognized="-", is_correct=False) for k in range(i2-i1)]
        elif tag == "insert":
            details += [PhonemeDetail(expected="-", recognized=hyp_chars[j1+k], is_correct=False) for k in range(j2-j1)]
    return details


# ── Service ───────────────────────────────────────────────────────────────────

class PronunciationService:
    def __init__(self):
        self._processor: Optional[Wav2Vec2Processor] = None
        self._model: Optional[Wav2Vec2ForCTC] = None

    def _load_model(self):
        if self._processor is None:
            logger.info("Đang tải mô hình %s ...", settings.MODEL_NAME)
            self._processor = Wav2Vec2Processor.from_pretrained(
                settings.MODEL_NAME, cache_dir=settings.MODEL_CACHE_DIR
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                settings.MODEL_NAME, cache_dir=settings.MODEL_CACHE_DIR
            )
            self._model.eval()
            logger.info("Tải mô hình thành công.")

    def _transcribe(self, audio_bytes: bytes) -> str:
        self._load_model()
        audio_array, _ = librosa.load(
            io.BytesIO(audio_bytes), sr=settings.SAMPLE_RATE, mono=True
        )
        max_samples = int(settings.MAX_AUDIO_DURATION_SEC * settings.SAMPLE_RATE)
        audio_array = audio_array[:max_samples]
        inputs = self._processor(
            audio_array, sampling_rate=settings.SAMPLE_RATE, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return self._processor.batch_decode(torch.argmax(logits, dim=-1))[0].lower().strip()

    def assess(
        self, audio_bytes: bytes, reference_text: str, include_phoneme_details: bool = False
    ) -> RecognizeResponse:
        transcription = self._transcribe(audio_bytes)
        cer = _compute_cer(reference_text, transcription)
        wer = _compute_wer(reference_text, transcription)
        score = _compute_score(cer, wer)
        level = _get_level(score)
        return RecognizeResponse(
            transcription=transcription,
            reference_text=reference_text,
            score=score,
            level=level,
            cer=round(cer, 4),
            wer=round(wer, 4),
            feedback=_get_feedback(level, cer, wer),
            phoneme_details=_build_phoneme_details(reference_text, transcription) if include_phoneme_details else None,
        )


pronunciation_service = PronunciationService()
