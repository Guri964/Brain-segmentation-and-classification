"""Model registry and checkpoint loading.

Responsibilities
----------------
* describe every model the system knows about (input specification, output
  semantics, architecture, file, expected parameters, provenance);
* load a checkpoint with the *correct* architecture and fail loudly rather than
  loading incompatible weights into a look-alike network;
* cache loaded modules so an HTTP request never pays the load cost twice;
* report honestly which models are usable and why the others are not.

Checkpoint formats handled
--------------------------
``legacy_pickle``   zip-less torch container produced by torch < 1.6
                    (``UNet-[16, 32, 64, 128, 256].pt``).
``zip``             modern torch.save container (``segnet_model.pth``,
                    ``radiomics_cnn_model.pth``), possibly storing tensors with a
                    ``cuda:N`` device tag, which is remapped to the local device.
``state_dict``      plain mapping of parameter name -> tensor
``wrapped``         ``{"state_dict": ..., "config": ..., "epoch": ...}`` produced
                    by the training scripts in this repository.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from ..utils.io import utc_now
from .architectures.attention_unet import RAUNet, RAUNetConfig, build_raunet
from .architectures.dynamic_unet import DEFAULT_FILTERS, DynamicUNet
from .architectures.segnet import SegNet

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InputSpec:
    channels: int
    height: int
    width: int
    normalisation: str
    dtype: str = "float32"

    @property
    def is_single_channel(self) -> bool:
        return self.channels == 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelDescriptor:
    model_id: str
    display_name: str
    task: str  # segmentation | classification
    architecture: str
    path: str
    input_spec: InputSpec
    output_semantics: str  # probability | logit | class_logits
    trainer: str  # which script produces a replacement checkpoint
    builder: Callable[[], nn.Module] | None = field(default=None, repr=False)
    fixed_threshold: float = 0.5
    description: str = ""
    supports_mc_dropout: bool = False
    provenance: str = ""
    expected_feature_names: list[str] | None = None
    requires_feature_schema: bool = False
    status_override: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task": self.task,
            "architecture": self.architecture,
            "path": self.path,
            "input_spec": self.input_spec.to_dict(),
            "output_semantics": self.output_semantics,
            "trainer": self.trainer,
            "fixed_threshold": self.fixed_threshold,
            "description": self.description,
            "supports_mc_dropout": self.supports_mc_dropout,
            "provenance": self.provenance,
            "requires_feature_schema": self.requires_feature_schema,
        }


@dataclass
class LoadedModel:
    descriptor: ModelDescriptor
    module: nn.Module | None
    device: torch.device
    status: str  # ready | incompatible | missing | unsupported_container | error
    message: str
    loaded_at: str | None = None
    file_size_bytes: int | None = None
    file_mtime: str | None = None
    parameter_count: int | None = None
    load_seconds: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and self.module is not None

    def to_dict(self) -> dict:
        payload = self.descriptor.to_dict()
        payload.update(
            {
                "status": self.status,
                "message": self.message,
                "device": str(self.device),
                "loaded_at": self.loaded_at,
                "file_size_bytes": self.file_size_bytes,
                "file_mtime": self.file_mtime,
                "parameter_count": self.parameter_count,
                "load_seconds": self.load_seconds,
                "available": self.is_ready,
            }
        )
        payload.update(self.extra)
        return payload


# --------------------------------------------------------------------------
# Device handling
# --------------------------------------------------------------------------
def resolve_device(preference: str = "auto") -> torch.device:
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(preference)
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        try:
            if torch.cuda.device_count() > 0:
                return torch.device("cuda:0")
        except Exception:  # pragma: no cover - driver issues
            pass
    return torch.device("cpu")


def apply_thread_configuration(threads: int) -> None:
    if threads and threads > 0:
        try:
            torch.set_num_threads(int(threads))
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
_CONTAINER_CACHE: dict[tuple[str, float], dict] = {}


class CheckpointFormatError(RuntimeError):
    """Raised when a checkpoint cannot be interpreted in any supported format."""


def inspect_checkpoint(path: Path) -> dict:
    """Return structural facts about a checkpoint file without loading tensors."""
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    with open(path, "rb") as handle:
        head = handle.read(4)
    legacy_magic = np.frombuffer(bytes.fromhex("1950a86a20f9469cfc6c"), dtype="<u4")[0]
    info: dict[str, Any] = {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "is_zip": head[:2] == b"PK",
    }
    if not info["is_zip"] and len(head) >= 4:
        magic = int.from_bytes(head, "little")
        if magic == int(legacy_magic):
            info["container"] = "legacy_pickle"
            with open(path, "rb") as handle:
                handle.read(2)
                prefix = handle.read(1)
                length = int.from_bytes(handle.read(8), "little")
                payload = handle.read(length)
            if payload[:2] == b"\x80\x02" or payload[:1] == b"c":
                info["requires_weights_only_false"] = True
        else:
            info["container"] = "unknown"
    elif info["is_zip"]:
        info["container"] = "zip"
    return info


def load_checkpoint_state(path: Path, device: torch.device | str = "cpu") -> tuple[dict, dict]:
    """Load a checkpoint and return ``(state_dict, metadata)``.

    The loader accepts the legacy container, modern zip containers and plain
    dictionaries.  Tensors tagged with a different device are mapped onto `device`,
    so the checkpoint only has to be loaded from disk once.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointFormatError(f"checkpoint not found: {path}")

    cache_key = (str(path), path.stat().st_mtime)
    if cache_key in _CONTAINER_CACHE:
        payload = _CONTAINER_CACHE[cache_key]
        return payload["state_dict"], payload["metadata"]

    metadata: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    obj: Any
    # torch >= 1.6 zip container / legacy pickle are both handled by torch.load;
    # weights_only=True is attempted first for the modern containers because it is
    # the safe path, and disabled only when the container is a legacy pickle.
    try:
        obj = torch.load(str(path), map_location=device, weights_only=True)
        metadata["load_mode"] = "weights_only"
    except Exception as weights_only_error:  # noqa: BLE001 - any failure falls back
        try:
            obj = torch.load(str(path), map_location=device, weights_only=False)
            metadata["load_mode"] = "full_pickle"
            metadata["weights_only_error"] = type(weights_only_error).__name__
        except Exception as exc:  # noqa: BLE001
            raise CheckpointFormatError(
                f"cannot deserialise {path.name}: {type(exc).__name__}: {exc}"
            ) from exc

    config = None
    epoch = None
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        config = obj.get("config")
        epoch = obj.get("epoch")
        obj = obj["state_dict"]
        metadata["wrapped"] = True

    if not isinstance(obj, dict):
        raise CheckpointFormatError(
            f"unsupported checkpoint root type {type(obj).__name__} in {path.name}"
        )

    state_dict = {}
    for key, value in obj.items():
        if isinstance(value, torch.nn.Parameter):
            value = value.detach()
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.clone().to(torch.device("cpu"))
        elif isinstance(value, (int, float, str, bool)):
            continue  # scalars such as epoch/global_step are not parameters
        else:
            continue
    if not state_dict:
        raise CheckpointFormatError(f"no tensors found in {path.name}")

    metadata["tensor_count"] = len(state_dict)
    metadata["config"] = config
    metadata["epoch"] = epoch
    _CONTAINER_CACHE[cache_key] = {"state_dict": state_dict, "metadata": metadata}
    return state_dict, metadata


def describe_state_dict(state_dict: dict) -> dict:
    """Summarise tensor shapes so a mismatch can be reported precisely."""
    shapes = {key: list(tensor.shape) for key, tensor in state_dict.items()}
    first_conv = next(
        (key for key in shapes if key.endswith("weight") and len(shapes[key]) == 4), None
    )
    return {
        "tensor_count": len(shapes),
        "first_conv_key": first_conv,
        "first_conv_shape": shapes[first_conv] if first_conv else None,
        "shapes": shapes,
    }


def load_into_module(
    module: nn.Module, state_dict: dict, *, strict: bool = False
) -> dict:
    """Load a state dict and report every missing/unexpected key.

    `strict=False` is used deliberately so that the caller receives a diagnostic
    report instead of an exception; the caller decides whether the mismatch is
    acceptable (it never is for inference, and the registry enforces that).
    """
    result = module.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "compatible": not result.missing_keys and not result.unexpected_keys,
    }


def tensor_stats(state_dict: dict, key: str) -> dict | None:
    if key not in state_dict:
        return None
    tensor = state_dict[key].float()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class ModelRegistry:
    """Holds descriptors, lazily loads checkpoints and caches the modules."""

    def __init__(self, config, *, device: torch.device | None = None):
        self.config = config
        self.device = device or resolve_device(getattr(config, "DEVICE_PREFERENCE", "auto"))
        apply_thread_configuration(getattr(config, "TORCH_THREADS", 0))
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._cache: dict[str, LoadedModel] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        paths = self.config.MODEL_PATHS

        self._descriptors["segnet_brats_4ch"] = ModelDescriptor(
            model_id="segnet_brats_4ch",
            display_name="SegNet (BraTS 4-channel)",
            task="segmentation",
            architecture="SegNet (2-stage encoder / bilinear decoder, no skip connections)",
            path=paths["segnet_brats_4ch"],
            input_spec=InputSpec(
                channels=4,
                height=256,
                width=256,
                normalisation="global z-score over the (H,W,C) block, then CHW",
            ),
            output_semantics="logit",
            trainer="scripts/train_segnet.py",
            builder=lambda: SegNet(in_channels=4, out_channels=1),
            description=(
                "Trained in the company notebook on BraTS 2020 four-channel slices "
                "(resized to 256x256) with a combination of Dice and BCE loss.  The "
                "checkpoint emits logits; the sigmoid is applied by the inference "
                "service."
            ),
            provenance="Copy_of_Ensemble_Framework_for_Brain_Abnormality.ipynb, cell 22/29",
        )

        self._descriptors["unet_dynamic_512"] = ModelDescriptor(
            model_id="unet_dynamic_512",
            display_name="U-Net (DynamicUNet, 512x512 grayscale)",
            task="segmentation",
            architecture="DynamicUNet, 4 encoder stages, filters [16,32,64,128,256], sigmoid output",
            path=paths["unet_dynamic_512"],
            input_spec=InputSpec(
                channels=1,
                height=512,
                width=512,
                normalisation="[0,1] intensity scaling (image / 255), no z-score",
            ),
            output_semantics="probability",
            trainer="scripts/train_unet.py --single-channel (or retrain on matched data)",
            builder=lambda: DynamicUNet(DEFAULT_FILTERS, input_channels=1, output_channels=1),
            description=(
                "Single-channel 512x512 U-Net reused from the existing repository.  It is "
                "the only model that can legitimately be applied to a plain grayscale "
                "MRI export uploaded through the interface."
            ),
            provenance="bts/model.py, final.py (existing Streamlit application)",
        )

        self._descriptors["raunet_brats_4ch"] = ModelDescriptor(
            model_id="raunet_brats_4ch",
            display_name="RAUNet (residual attention U-Net, deep supervision)",
            task="segmentation",
            architecture=(
                "Residual attention U-Net: 4 encoder stages [16,32,64,128,256], additive "
                "attention gates on every skip connection, deep supervision heads"
            ),
            path=paths["raunet_brats_4ch"],
            input_spec=InputSpec(
                channels=4,
                height=256,
                width=256,
                normalisation="global z-score over the (H,W,C) block, then CHW (identical to SegNet)",
            ),
            output_semantics="logit",
            trainer="scripts/train_unet.py",
            builder=lambda: build_raunet(in_channels=4, out_channels=1, base_filters=16, depth=4),
            description=(
                "Research model implemented for this project.  Input-compatible with the "
                "reused SegNet checkpoint, which makes a probability-level ensemble "
                "possible.  Trained by scripts/train_unet.py; the checkpoint is written "
                "to data/models/ and is absent until training has been run."
            ),
            supports_mc_dropout=True,
            provenance="implemented in backend/app/ml/architectures/attention_unet.py",
        )

        self._descriptors["radiomics_mlp_v1"] = ModelDescriptor(
            model_id="radiomics_mlp_v1",
            display_name="Radiomics MLP v1 (incompatible)",
            task="classification",
            architecture="TumorClassifier: Linear(21->128) -> ReLU -> Linear(128->64) -> ReLU -> Linear(64->2)",
            path=paths["radiomics_mlp_v1"],
            input_spec=InputSpec(
                channels=21,
                height=1,
                width=1,
                normalisation="none recorded (NaNs filled with 0 before training)",
            ),
            output_semantics="class_logits",
            trainer="scripts/train_classifier.py",
            builder=None,
            requires_feature_schema=True,
            description=(
                "Checkpoint produced by the existing notebook.  It expects 21 radiomics "
                "features but the repository only ships features_df.csv with 11 columns, "
                "and the label vector used during training is undefined in the notebook. "
                "No feature schema, scaler, split or metric artefact was saved.  The "
                "classifier is therefore reported as incompatible and stays disabled."
            ),
            provenance="radiomics_cnn_model.pth (notebook cell 40)",
        )

    # ------------------------------------------------------------------
    def get_descriptor(self, model_id: str) -> ModelDescriptor:
        if model_id not in self._descriptors:
            raise KeyError(model_id)
        return self._descriptors[model_id]

    def register(self, descriptor: ModelDescriptor) -> None:
        self._descriptors[descriptor.model_id] = descriptor

    @property
    def descriptors(self) -> dict[str, ModelDescriptor]:
        return dict(self._descriptors)

    def segmentation_model_ids(self) -> list[str]:
        return [k for k, d in self._descriptors.items() if d.task == "segmentation"]

    def available_segmentation_models(self) -> list[LoadedModel]:
        return [
            model
            for model in (self.load_model(mid) for mid in self.segmentation_model_ids())
            if model.is_ready
        ]

    # ------------------------------------------------------------------
    def load_model(self, model_id: str, *, force: bool = False) -> LoadedModel:
        """Load (or fetch from cache) the checkpoint described by `model_id`."""
        if model_id in self._cache and not force:
            return self._cache[model_id]

        descriptor = self.get_descriptor(model_id)
        path = Path(descriptor.path)
        started = time.perf_counter()

        if descriptor.requires_feature_schema:
            loaded = LoadedModel(
                descriptor=descriptor,
                module=None,
                device=self.device,
                status="incompatible",
                message=(
                    "Feature schema required by this checkpoint is unavailable; the shipped "
                    "feature table has 11 columns while the network expects 21 inputs."
                ),
                file_size_bytes=path.stat().st_size if path.exists() else None,
            )
            self._cache[model_id] = loaded
            return loaded

        if not path.exists():
            loaded = LoadedModel(
                descriptor=descriptor,
                module=None,
                device=self.device,
                status="missing",
                message=(
                    f"Checkpoint not found at {path}. "
                    f"Train it with {descriptor.trainer}."
                ),
            )
            self._cache[model_id] = loaded
            return loaded

        if descriptor.builder is None:
            loaded = LoadedModel(
                descriptor=descriptor,
                module=None,
                device=self.device,
                status="unsupported_container",
                message="No architecture builder is registered for this model.",
                file_size_bytes=path.stat().st_size,
            )
            self._cache[model_id] = loaded
            return loaded

        try:
            state_dict, metadata = load_checkpoint_state(path, self.device)
            module = descriptor.builder()
            report = load_into_module(module, state_dict, strict=False)
            if not report["compatible"]:
                loaded = LoadedModel(
                    descriptor=descriptor,
                    module=None,
                    device=self.device,
                    status="incompatible",
                    message=(
                        "Checkpoint does not match the registered architecture: "
                        f"{len(report['missing_keys'])} missing and "
                        f"{len(report['unexpected_keys'])} unexpected parameters."
                    ),
                    file_size_bytes=path.stat().st_size,
                    extra={
                        "missing_keys": report["missing_keys"][:20],
                        "unexpected_keys": report["unexpected_keys"][:20],
                        "checkpoint_container": metadata.get("container", "zip"),
                    },
                )
                self._cache[model_id] = loaded
                logger.error("Model %s incompatible: %s", model_id, loaded.message)
                return loaded

            module.eval()
            module.to(self.device)
            parameter_count = int(sum(p.numel() for p in module.parameters()))
            loaded = LoadedModel(
                descriptor=descriptor,
                module=module,
                device=self.device,
                status="ready",
                message="Checkpoint loaded and verified.",
                loaded_at=utc_now(),
                file_size_bytes=path.stat().st_size,
                file_mtime=utc_now(),
                parameter_count=parameter_count,
                load_seconds=round(time.perf_counter() - started, 4),
                extra={
                    "checkpoint_container": metadata.get("container", "zip"),
                    "load_mode": metadata.get("load_mode"),
                    "tensor_count": metadata.get("tensor_count"),
                },
            )
            logger.info(
                "Loaded %s (%s parameters, %.2fs)", model_id, parameter_count, loaded.load_seconds
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a status
            loaded = LoadedModel(
                descriptor=descriptor,
                module=None,
                device=self.device,
                status="error",
                message=f"{type(exc).__name__}: {exc}",
                file_size_bytes=path.stat().st_size if path.exists() else None,
            )
            logger.exception("Failed to load %s", model_id)

        self._cache[model_id] = loaded
        return loaded

    # ------------------------------------------------------------------
    def preload(self, model_ids: list[str] | None = None) -> dict[str, LoadedModel]:
        model_ids = model_ids or list(self._descriptors)
        return {model_id: self.load_model(model_id) for model_id in model_ids}

    def status_report(self) -> list[dict]:
        return [self.load_model(model_id).to_dict() for model_id in self._descriptors]

    def summary(self) -> dict:
        statuses = self.status_report()
        ready = [entry for entry in statuses if entry["available"]]
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "models": statuses,
            "ready_count": len(ready),
            "total_count": len(statuses),
            "generated_at": utc_now(),
        }

    def input_compat_groups(self) -> list[dict]:
        """Group ready segmentation models by identical input specification."""
        groups: dict[tuple, list[str]] = {}
        for model in self.available_segmentation_models():
            spec = model.descriptor.input_spec
            key = (spec.channels, spec.height, spec.width, spec.normalisation)
            groups.setdefault(key, []).append(model.descriptor.model_id)
        return [
            {
                "channels": key[0],
                "height": key[1],
                "width": key[2],
                "normalisation": key[3],
                "model_ids": value,
                "ensemblable": len(value) > 1,
            }
            for key, value in groups.items()
        ]
