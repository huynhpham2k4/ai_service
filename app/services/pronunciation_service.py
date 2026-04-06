import io
import logging
import difflib
from typing import Optional

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.core.config import get_settings
from app.schemas.recognize import PhonemeDetail, PronunciationLevel, RecognizeResponse

logger = logging.getLogger(__name__)
settings = get_settings()


def _compute_cer(reference: str, hypothesis: str) -> float:
    """
    Tính Character Error Rate (CER) giữa reference và hypothesis.
    CER = (S + D + I) / N  với N là số ký tự trong reference.
    """
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Levenshtein distance via dynamic programming
    dp = list(range(len(hyp) + 1))
    for i, r_char in enumerate(ref):
        new_dp = [i + 1]
        for j, h_char in enumerate(hyp):
            if r_char == h_char:
                new_dp.append(dp[j])
            else:
                new_dp.append(1 + min(dp[j], dp[j + 1], new_dp[-1]))
        dp = new_dp

    return dp[len(hyp)] / len(ref)


def _compute_wer(reference: str, hypothesis: str) -> float:
    """
    Tính Word Error Rate (WER) giữa reference và hypothesis.
    WER = (S + D + I) / N  với N là số từ trong reference.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dp = list(range(len(hyp_words) + 1))
    for i, r_word in enumerate(ref_words):
        new_dp = [i + 1]
        for j, h_word in enumerate(hyp_words):
            if r_word == h_word:
                new_dp.append(dp[j])
            else:
                new_dp.append(1 + min(dp[j], dp[j + 1], new_dp[-1]))
        dp = new_dp

    return dp[len(hyp_words)] / len(ref_words)


def _compute_score(cer: float, wer: float) -> float:
    """
    Chuyển CER và WER thành điểm số từ 0-100.
    Công thức: score = (1 - 0.6 * WER - 0.4 * CER) * 100, clamp về [0, 100].
    """
    score = (1.0 - 0.6 * wer - 0.4 * cer) * 100.0
    return round(max(0.0, min(100.0, score)), 2)


def _get_level(score: float) -> PronunciationLevel:
    if score >= settings.SCORE_EXCELLENT:
        return PronunciationLevel.EXCELLENT
    elif score >= settings.SCORE_GOOD:
        return PronunciationLevel.GOOD
    elif score >= settings.SCORE_FAIR:
        return PronunciationLevel.FAIR
    return PronunciationLevel.POOR


def _get_feedback(level: PronunciationLevel, cer: float, wer: float) -> str:
    feedback_map = {
        PronunciationLevel.EXCELLENT: "Phát âm rất tốt! Tiếp tục phát huy.",
        PronunciationLevel.GOOD: "Phát âm tốt, chỉ có một số lỗi nhỏ.",
        PronunciationLevel.FAIR: f"Phát âm trung bình. CER={cer:.2%}, WER={wer:.2%}. Cần luyện tập thêm.",
        PronunciationLevel.POOR: f"Phát âm cần cải thiện. CER={cer:.2%}, WER={wer:.2%}. Hãy nghe lại và luyện tập.",
    }
    return feedback_map[level]


def _build_phoneme_details(reference: str, hypothesis: str) -> list[PhonemeDetail]:
    """So sánh từng ký tự giữa reference và hypothesis."""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    matcher = difflib.SequenceMatcher(None, ref_chars, hyp_chars)
    details: list[PhonemeDetail] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                details.append(
                    PhonemeDetail(
                        expected=ref_chars[i1 + k],
                        recognized=hyp_chars[j1 + k],
                        is_correct=True,
                    )
                )
        elif tag == "replace":
            len_ref = i2 - i1
            len_hyp = j2 - j1
            for k in range(max(len_ref, len_hyp)):
                expected = ref_chars[i1 + k] if k < len_ref else "-"
                recognized = hyp_chars[j1 + k] if k < len_hyp else "-"
                details.append(
                    PhonemeDetail(expected=expected, recognized=recognized, is_correct=False)
                )
        elif tag == "delete":
            for k in range(i2 - i1):
                details.append(
                    PhonemeDetail(expected=ref_chars[i1 + k], recognized="-", is_correct=False)
                )
        elif tag == "insert":
            for k in range(j2 - j1):
                details.append(
                    PhonemeDetail(expected="-", recognized=hyp_chars[j1 + k], is_correct=False)
                )

    return details


class PronunciationService:
    """
    Service đánh giá phát âm sử dụng mô hình Wav2Vec2.
    Mô hình được tải lazy khi gọi lần đầu và cache lại cho các lần sau.
    """

    def __init__(self):
        self._processor: Optional[Wav2Vec2Processor] = None
        self._model: Optional[Wav2Vec2ForCTC] = None

    def _load_model(self):
        if self._processor is None or self._model is None:
            logger.info("Đang tải mô hình Wav2Vec2: %s ...", settings.MODEL_NAME)
            self._processor = Wav2Vec2Processor.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
            )
            self._model.eval()
            logger.info("Tải mô hình thành công.")

    def _transcribe(self, audio_bytes: bytes) -> str:
        """Nhận dạng giọng nói từ bytes âm thanh, trả về chuỗi văn bản."""
        self._load_model()

        # Đọc audio từ bytes, resample về SAMPLE_RATE nếu cần
        audio_array, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=settings.SAMPLE_RATE,
            mono=True,
        )

        # Giới hạn độ dài âm thanh
        max_samples = int(settings.MAX_AUDIO_DURATION_SEC * settings.SAMPLE_RATE)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        inputs = self._processor(
            audio_array,
            sampling_rate=settings.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription: str = self._processor.batch_decode(predicted_ids)[0]
        return transcription.lower().strip()

    def assess(
        self,
        audio_bytes: bytes,
        reference_text: str,
        include_phoneme_details: bool = False,
    ) -> RecognizeResponse:
        """
        Đánh giá phát âm.

        Args:
            audio_bytes: Nội dung file âm thanh (bytes).
            reference_text: Từ / câu người dùng muốn phát âm.
            include_phoneme_details: Trả về chi tiết từng âm vị hay không.

        Returns:
            RecognizeResponse chứa điểm số và nhận xét.
        """
        transcription = self._transcribe(audio_bytes)

        cer = _compute_cer(reference_text, transcription)
        wer = _compute_wer(reference_text, transcription)
        score = _compute_score(cer, wer)
        level = _get_level(score)
        feedback = _get_feedback(level, cer, wer)

        phoneme_details = (
            _build_phoneme_details(reference_text, transcription)
            if include_phoneme_details
            else None
        )

        return RecognizeResponse(
            transcription=transcription,
            reference_text=reference_text,
            score=score,
            level=level,
            cer=round(cer, 4),
            wer=round(wer, 4),
            feedback=feedback,
            phoneme_details=phoneme_details,
        )


# Singleton instance — dùng chung trong toàn ứng dụng
pronunciation_service = PronunciationService()
