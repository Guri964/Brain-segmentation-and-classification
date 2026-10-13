from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(BACKEND_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
@pytest.fixture(scope="session")
def app_config(tmp_path_factory):
    import os
    data_dir = tmp_path_factory.mktemp("brain-data")
    os.environ["BRAIN_ENV"] = "testing"
    os.environ["BRAIN_DATA_DIR"] = str(data_dir)
    from app.config import TestingConfig
    class SandboxConfig(TestingConfig):
        DATA_DIR = data_dir
        LOG_DIR = data_dir / "logs"
    SandboxConfig.init_directories()
    return SandboxConfig
@pytest.fixture(scope="session")
def app(app_config):
    from app import create_app
    application = create_app(config_object=app_config, env="testing")
    application.config.update(TESTING=True)
    return application
@pytest.fixture()
def client(app):
    return app.test_client()
@pytest.fixture()
def services(app):
    return app.extensions["brain_services"]
@pytest.fixture()
def brain_image() -> np.ndarray:
    size = 256
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = size / 2
    skull = np.exp(-(((yy - centre) ** 2 + (xx - centre) ** 2) / (2 * (size * 0.42) ** 2)))
    lesion = np.exp(-(((yy - size * 0.42) ** 2 + (xx - size * 0.58) ** 2) / (2 * (size * 0.07) ** 2)))
    texture = 0.05 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    return np.clip(skull * 0.7 + lesion + texture, 0.0, 1.0).astype(np.float32)
@pytest.fixture()
def brain_png(brain_image, tmp_path) -> Path:
    from PIL import Image
    path = tmp_path / "phantom.png"
    Image.fromarray((brain_image * 255).astype(np.uint8)).save(path)
    return path
@pytest.fixture()
def four_channel_pngs(brain_image, tmp_path) -> list[Path]:
    from PIL import Image
    paths = []
    for index, (offset, scale) in enumerate([(0.0, 1.0), (0.05, 1.1), (0.12, 0.9), (0.2, 1.25)]):
        shifted = np.roll(brain_image, int(brain_image.shape[0] * offset), axis=index % 2)
        array = np.clip(shifted * scale, 0.0, 1.0)
        path = tmp_path / f"modality_{index}.png"
        Image.fromarray((array * 255).astype(np.uint8)).save(path)
        paths.append(path)
    return paths
@pytest.fixture()
def reference_mask(brain_image) -> np.ndarray:
    return (brain_image > 0.8).astype(np.float32)
