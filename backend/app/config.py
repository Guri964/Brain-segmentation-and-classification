from __future__ import annotations
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default
class BaseConfig:
    ENV_NAME = "base"
    DEBUG = False
    TESTING = False
    DATA_DIR = Path(os.environ.get("BRAIN_DATA_DIR", BASE_DIR / "data"))
    UPLOAD_DIR_NAME = "uploads"
    ARTIFACT_DIR_NAME = "artifacts"
    REPORT_DIR_NAME = "reports"
    MODEL_DIR_NAME = "models"
    DATABASE_FILENAME = "app.db"
    MAX_CONTENT_LENGTH = _env_int("BRAIN_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)
    MAX_FILES_PER_ANALYSIS = _env_int("BRAIN_MAX_FILES_PER_ANALYSIS", 8)
    JSON_SORT_KEYS = False
    CORS_ALLOWED_ORIGINS = os.environ.get(
        "BRAIN_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    DEVICE_PREFERENCE = os.environ.get("BRAIN_DEVICE", "auto")
    TORCH_THREADS = _env_int("BRAIN_TORCH_THREADS", 2)
    MODEL_CACHE_ENABLED = _env_bool("BRAIN_MODEL_CACHE", True)
    ENABLE_TTA_BY_DEFAULT = _env_bool("BRAIN_ENABLE_TTA", False)
    MC_DROPOUT_SAMPLES = _env_int("BRAIN_MC_DROPOUT_SAMPLES", 8)
    SEGMENTATION_THRESHOLD = _env_float("BRAIN_SEG_THRESHOLD", 0.5)
    MODEL_PATHS = {
        "segnet_brats_4ch": os.environ.get(
            "BRAIN_MODEL_SEGNET", str(BASE_DIR / "segnet_model.pth")
        ),
        "unet_dynamic_512": os.environ.get(
            "BRAIN_MODEL_UNET", str(BASE_DIR / "UNet-[16, 32, 64, 128, 256].pt")
        ),
        "raunet_brats_4ch": os.environ.get(
            "BRAIN_MODEL_RAUNET", str(BASE_DIR / "data" / "models" / "raunet_brats_4ch.pt")
        ),
        "radiomics_mlp_v1": os.environ.get(
            "BRAIN_MODEL_RADIOMICS_MLP", str(BASE_DIR / "radiomics_cnn_model.pth")
        ),
    }
    ENSEMBLE_CALIBRATION_PATH = os.environ.get(
        "BRAIN_ENSEMBLE_CALIBRATION",
        str(BASE_DIR / "data" / "models" / "ensemble_calibration.json"),
    )
    CLASSIFIER_ARTIFACT_DIR = os.environ.get(
        "BRAIN_CLASSIFIER_DIR", str(BASE_DIR / "data" / "models" / "classification")
    )
    RADIOMICS_BACKEND = os.environ.get("BRAIN_RADIOMICS_BACKEND", "auto")
    RADIOMICS_BIN_COUNT = _env_int("BRAIN_RADIOMICS_BIN_COUNT", 32)
    RADIOMICS_BIN_WIDTH = _env_float("BRAIN_RADIOMICS_BIN_WIDTH", 0.0)
    RADIOMICS_DISCRETIZATION = os.environ.get("BRAIN_RADIOMICS_DISCRETIZATION", "bin_count")
    ASSISTANT_LLM_ENABLED = _env_bool("BRAIN_ASSISTANT_LLM_ENABLED", False)
    ASSISTANT_LLM_API_KEY = os.environ.get("BRAIN_ASSISTANT_LLM_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    ASSISTANT_LLM_MODEL = os.environ.get("BRAIN_ASSISTANT_LLM_MODEL", "gpt-4o-mini")
    ASSISTANT_LLM_BASE_URL = os.environ.get("BRAIN_ASSISTANT_LLM_BASE_URL", "")
    LOG_LEVEL = os.environ.get("BRAIN_LOG_LEVEL", "INFO")
    LOG_DIR = Path(os.environ.get("BRAIN_LOG_DIR", BASE_DIR / "data" / "logs"))
    DISCLAIMER = (
        "Research prototype. Outputs are AI-generated and are not intended to replace "
        "professional medical diagnosis."
    )
    @classmethod
    def init_directories(cls) -> None:
        for path in (
            cls.DATA_DIR,
            cls.DATA_DIR / cls.UPLOAD_DIR_NAME,
            cls.DATA_DIR / cls.ARTIFACT_DIR_NAME,
            cls.DATA_DIR / cls.REPORT_DIR_NAME,
            cls.DATA_DIR / cls.MODEL_DIR_NAME,
            cls.LOG_DIR,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)
    @classmethod
    def database_path(cls) -> Path:
        return Path(cls.DATA_DIR) / cls.DATABASE_FILENAME
    @classmethod
    def as_dict(cls) -> dict:
        return {
            "env": cls.ENV_NAME,
            "device_preference": cls.DEVICE_PREFERENCE,
            "model_paths": cls.MODEL_PATHS,
            "radiomics_backend": cls.RADIOMICS_BACKEND,
            "max_upload_bytes": cls.MAX_CONTENT_LENGTH,
            "assistant_llm_enabled": cls.ASSISTANT_LLM_ENABLED,
        }
class DevelopmentConfig(BaseConfig):
    ENV_NAME = "development"
    DEBUG = True
class TestingConfig(BaseConfig):
    ENV_NAME = "testing"
    TESTING = True
    DEBUG = False
    TORCH_THREADS = 1
    MODEL_CACHE_ENABLED = False
    DATA_DIR = Path(os.environ.get("BRAIN_DATA_DIR", "/tmp/brain_test_data"))
class ProductionConfig(BaseConfig):
    ENV_NAME = "production"
    DEBUG = False
    MODEL_CACHE_ENABLED = True
    LOG_LEVEL = os.environ.get("BRAIN_LOG_LEVEL", "WARNING")
_CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
def get_config(env: str | None = None):
    env = (env or os.environ.get("BRAIN_ENV") or "development").lower()
    return _CONFIGS.get(env, DevelopmentConfig)
