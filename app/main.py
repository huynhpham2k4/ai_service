import logging
import sys

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.recognize import router as recognize_router
from app.core.config import get_settings

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Khởi động %s v%s ...", settings.APP_NAME, settings.APP_VERSION)
    yield
    logger.info("Tắt ứng dụng.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "API đánh giá phát âm tiếng Anh sử dụng mô hình Wav2Vec2.\n\n"
        "Gửi file âm thanh cùng với từ / câu tham chiếu để nhận điểm phát âm."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Cấu hình lại trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(recognize_router, tags=["Pronunciation Assessment"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Kiểm tra trạng thái API")
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}
