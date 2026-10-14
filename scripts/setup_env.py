from __future__ import annotations
import argparse
import importlib
import platform
import subprocess
import sys
REQUIRED = [
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("torch", "torch"),
    ("flask", "flask"),
    ("flask_cors", "flask-cors"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("reportlab", "reportlab"),
]
OPTIONAL = [
    ("SimpleITK", "SimpleITK", "reading NIfTI / NRRD / DICOM volumes"),
    ("pydicom", "pydicom", "reading DICOM series"),
    ("radiomics", "pyradiomics", "the reference radiomics extractor"),
    ("shap", "shap", "SHAP-style feature attribution instead of the linear surrogate"),
    ("psutil", "psutil", "memory reporting in the evaluation output"),
]
def status(module: str) -> str | None:
    try:
        loaded = importlib.import_module(module)
    except Exception:
        return None
    return getattr(loaded, "__version__", "installed")
def main() -> int:
    parser = argparse.ArgumentParser(description="Report and optionally install the runtime dependencies")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--torch-index-url", default=None, help="Passed to pip, e.g. the CPU-only wheel index")
    parser.add_argument("--skip-torch", action="store_true", help="Do not touch torch (useful when it is already pinned)")
    args = parser.parse_args()
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print("")
    missing: list[str] = []
    print("Required")
    for module, package in REQUIRED:
        version = status(module)
        print(f"  {'OK ' if version else 'MISSING'} {package:<18} {version or '-'}")
        if not version and not (args.skip_torch and package == "torch"):
            missing.append(package)
    print("")
    print("Optional (a feature degrades gracefully without it, and the API says which)")
    for module, package, why in OPTIONAL:
        version = status(module)
        note = version or f"- {why}"
        if package == "pyradiomics" and not version and sys.version_info >= (3, 11):
            note = f"- no wheel for Python {sys.version_info.major}.{sys.version_info.minor}; the native extractor is used"
        print(f"  {'OK ' if version else 'MISSING'} {package:<18} {note}")
    print("")
    try:
        import torch
        print(f"torch {torch.__version__} · CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("  CPU execution: expect roughly 0.5 s per 512x512 single-channel image.")
    except Exception:
        print("torch is not importable yet.")
    if not args.install:
        if missing:
            print("")
            print("Re-run with --install to install the missing packages:")
            print("  " + " ".join(missing))
        return 0
    if missing:
        command = [sys.executable, "-m", "pip", "install", *missing]
        if args.torch_index_url and "torch" in missing:
            command += ["--index-url", args.torch_index_url]
        print("")
        print("Running: " + " ".join(command))
        subprocess.check_call(command)
    else:
        print("")
        print("Nothing to install.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
