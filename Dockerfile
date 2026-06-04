FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by soundfile, librosa, and building some python modules
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Warm up / download HuggingFace models and NLTK datasets during build stage.
# We preload both the default model (from config.py) and the example model (from .env.example)
# to ensure the container can start instantly offline/without downloading at runtime.
RUN python -c " \
import nltk; \
nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
nltk.download('cmudict', quiet=True); \
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
for m in ['vitouphy/wav2vec2-xls-r-300m-phoneme', 'facebook/wav2vec2-base-960h']: \
    try: \
        Wav2Vec2Processor.from_pretrained(m, cache_dir='./model_cache'); \
        Wav2Vec2ForCTC.from_pretrained(m, cache_dir='./model_cache'); \
        print(f'Successfully pre-loaded model: {m}'); \
    except Exception as e: \
        print(f'Could not pre-load model {m}: {e}'); \
from g2p_en import G2p; \
g2p = G2p() \
"

EXPOSE 8000

ENV DEBUG=false
ENV MODEL_CACHE_DIR=./model_cache

# Start uvicorn without reload for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
