from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_NAME: str = "vitouphy/wav2vec2-xls-r-300m-phoneme"
    MODEL_CACHE_DIR: str = "./model_cache"
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_DURATION_SEC: float = 30.0
    MAX_FILE_SIZE_MB: float = 10.0

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
