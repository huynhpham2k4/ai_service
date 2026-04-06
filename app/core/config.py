from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "Pronunciation Assessment API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Wav2Vec2 model config
    MODEL_NAME: str = "facebook/wav2vec2-base-960h"
    MODEL_CACHE_DIR: str = "./model_cache"

    # Audio config
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_DURATION_SEC: float = 30.0
    MAX_FILE_SIZE_MB: float = 10.0

    # Scoring thresholds
    SCORE_EXCELLENT: float = 85.0
    SCORE_GOOD: float = 70.0
    SCORE_FAIR: float = 50.0

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
