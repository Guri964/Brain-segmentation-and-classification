"""The stacked ensemble of ``FINAL ENSEMBLE NB.ipynb``.

Everything here is driven by a temporary weights directory holding checkpoints that were written
by the *real* notebook architecture classes (`.state_dict()` of
``app.ml.architectures.notebook_ensemble``), the file names the notebook's artefacts ZIP uses,
and a feature schema/ scaler/meta-learner fitted exactly the way the notebook fits them.  That
exercises the whole serving path - paths, contracts, feature order, scaling, meta matrix, HTTP -
against fixtures.  No test in this file measures or claims accuracy: the numbers a model would
produce here are meaningless because the checkpoints contain random weights, and the notebook's
own metrics are asserted only as *reported*, never as re-measured.

Tests whose fixtures need MONAI (the notebook's U-Net member) skip with the install command when
MONAI is absent, because that member is one of the ten meta features.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.errors import ModelUnavailableError, ValidationError
from app.ml import radiomics_notebook
from app.ml.architectures.notebook_ensemble import (
    NotebookRAUNet,
    NotebookRadiomicsCNN,
    NotebookSegNet,
    monai_available,
)
from app.ml.registry import ModelRegistry
from app.services.ensemble_bundle_service import (
    META_FEATURE_COUNT,
    SEGMENTATION_MEMBERS,
    EnsembleBundleService,
    _TensorWrapper,
)
from app.services.preprocessing_service import (
    NORMALISATION_GLOBAL,
    NORMALISATION_PER_CHANNEL,
    preprocess_four_channel,
)

NOTEBOOK_MEMBER_FILES = {
    "segnet_ensemble_240": "segnet.pth",
    "unet_ensemble_240": "unet_best.pth",
    "raunet_ensemble_240": "raunet.pth",
}


# --------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def notebook_weights(tmp_path_factory) -> Path:
    """The eight artefacts the notebook's ZIP contains, written with real architectures."""

    directory = tmp_path_factory.mktemp("notebook-artifacts")
    torch.manual_seed(0)

    torch.save(NotebookSegNet().state_dict(), directory / "segnet.pth")
    torch.save(NotebookRAUNet().state_dict(), directory / "raunet.pth")
    torch.save(NotebookRadiomicsCNN(46).state_dict(), directory / "radiomics_cnn.pth")
    if monai_available():
        from app.ml.architectures.notebook_ensemble import build_monai_unet

        torch.save(build_monai_unet().state_dict(), directory / "unet_best.pth")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    import joblib

    rng = np.random.default_rng(0)
    # the schema the notebook writes: a DataFrame of extracted features, so the header row is the
    # feature order (`feat_df.to_csv(..., index=False)`)
    order = radiomics_notebook.feature_order()
    import csv

    with (directory / "radiomics_feature_names.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(order)
        for _ in range(3):
            writer.writerow([0.0] * len(order))

    feature_matrix = rng.normal(size=(64, len(order))).astype(np.float64)
    joblib.dump(
        StandardScaler().fit(feature_matrix), directory / "radiomics_scaler.pkl"
    )

    meta_matrix = rng.normal(size=(64, META_FEATURE_COUNT)).astype(np.float64)
    meta_labels = (meta_matrix[:, 0] > 0).astype(int)
    meta_scaler = StandardScaler().fit(meta_matrix)
    joblib.dump(meta_scaler, directory / "meta_scaler.pkl")
    learner = GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=42)
    learner.fit(meta_scaler.transform(meta_matrix), meta_labels)
    joblib.dump(learner, directory / "meta_learner.pkl")
    return directory


@pytest.fixture()
def notebook_config(notebook_weights):
    """A config whose ensemble artefacts all point into the temporary directory."""

    from app.config import TestingConfig

    class NotebookConfig(TestingConfig):
        MODEL_PATHS = {
            **TestingConfig.MODEL_PATHS,
            **{
                model_id: str(notebook_weights / name)
                for model_id, name in NOTEBOOK_MEMBER_FILES.items()
            },
            "radiomics_ensemble_cnn": str(notebook_weights / "radiomics_cnn.pth"),
        }
        RADIOMICS_ENSEMBLE_SCALER_PATH = str(notebook_weights / "radiomics_scaler.pkl")
        RADIOMICS_ENSEMBLE_SCHEMA_PATH = str(notebook_weights / "radiomics_feature_names.csv")
        META_LEARNER_PATH = str(notebook_weights / "meta_learner.pkl")
        META_SCALER_PATH = str(notebook_weights / "meta_scaler.pkl")

    return NotebookConfig


def notebook_tensor(seed: int = 0, size: int = 240):
    """A synthetic four-channel slice with a bright region, z-scored per channel."""

    rng = np.random.default_rng(seed)
    image = rng.normal(size=(4, size, size)).astype(np.float32)
    image[0, size // 3 : size // 2, size // 3 : size // 2] += 3.0
    for channel in range(4):
        plane = image[channel]
        image[channel] = (plane - plane.mean()) / (plane.std() + 1e-6)
    return torch.from_numpy(image)[None]


@pytest.fixture()
def services_for_notebook(notebook_config):
    from app.services.segmentation_service import SegmentationService

    registry = ModelRegistry(notebook_config)
    segmentation = SegmentationService(registry, notebook_config)
    bundle = EnsembleBundleService(notebook_config, registry, segmentation)
    return registry, segmentation, bundle


# ------------------------------------------------------------------ the feature extractor
def test_feature_order_is_the_notebook_order():
    order = radiomics_notebook.feature_order()
    assert len(order) == radiomics_notebook.FEATURE_COUNT == 46
    assert order[:11] == [
        "ch0_mean",
        "ch0_std",
        "ch0_skew",
        "ch0_kurt",
        "ch0_energy",
        "ch0_entropy",
        "ch0_p10",
        "ch0_p25",
        "ch0_p75",
        "ch0_p90",
        "ch0_range",
    ]
    assert order[11:13][0].startswith("ch1_")
    assert order[-2:] == ["tumor_size", "tumor_ratio"]


def test_statistics_match_scipy_the_library_the_notebook_used():
    from scipy.stats import entropy as scipy_entropy
    from scipy.stats import kurtosis as scipy_kurtosis
    from scipy.stats import skew as scipy_skew

    rng = np.random.default_rng(3)
    for _ in range(50):
        values = rng.normal(size=int(rng.integers(20, 800)))
        assert radiomics_notebook._skew(values) == pytest.approx(float(scipy_skew(values)), abs=1e-9)
        assert radiomics_notebook._kurtosis(values) == pytest.approx(
            float(scipy_kurtosis(values)), abs=1e-9
        )
        histogram, _ = np.histogram(values, bins=32, density=True)
        histogram = histogram + 1e-9
        assert radiomics_notebook._entropy(histogram) == pytest.approx(
            float(scipy_entropy(histogram)), abs=1e-12
        )


def test_a_constant_region_does_not_produce_a_nan():
    """A flat ROI divides by a zero variance in SciPy; the extractor must stay finite."""

    features = radiomics_notebook.extract_features(
        np.full((32, 32, 4), 2.0, dtype=np.float32), np.ones((32, 32), np.float32)
    )
    assert features is not None
    assert all(np.isfinite(value) for value in features.values())
    assert features["ch0_skew"] == 0.0 and features["ch0_kurt"] == 0.0


def test_an_empty_mask_returns_none_exactly_like_the_notebook():
    image = np.zeros((16, 16, 4), dtype=np.float32)
    assert radiomics_notebook.extract_features(image, np.zeros((16, 16), np.float32)) is None
    assert radiomics_notebook.extract_features(image, np.zeros((16, 16, 1), np.float32)) is None


def test_feature_values_are_the_ones_the_notebook_would_compute():
    rng = np.random.default_rng(5)
    image = rng.normal(size=(24, 24, 4)).astype(np.float32)
    mask = np.zeros((24, 24), np.float32)
    mask[6:18, 8:20] = 1
    features = radiomics_notebook.extract_features(image, mask)
    roi = image[..., 0][mask > 0]
    assert features["ch0_mean"] == pytest.approx(float(roi.mean()), abs=1e-6)
    assert features["ch0_std"] == pytest.approx(float(roi.std()), abs=1e-6)
    assert features["ch0_range"] == pytest.approx(float(roi.max() - roi.min()), abs=1e-6)
    assert features["tumor_size"] == 144.0
    assert features["ch0_p10"] == pytest.approx(float(np.percentile(roi, 10)), abs=1e-5)


def test_vector_refuses_a_missing_feature_instead_of_zero_filling():
    partial = {"tumor_size": 1.0}
    with pytest.raises(ValueError) as error:
        radiomics_notebook.vector(partial, radiomics_notebook.feature_order())
    assert "ch0_mean" in str(error.value)


def test_target_semantics_say_the_label_is_not_a_grade():
    text = radiomics_notebook.TARGET_SEMANTICS.lower()
    assert "tumor-size proxy" in text or "tumour-size proxy" in text
    assert "not the who grade" in text
    assert "never" in radiomics_notebook.LEAKAGE_NOTE.lower() or "leakage" in (
        radiomics_notebook.LEAKAGE_NOTE.lower()
    )


# ------------------------------------------------------------------------- the architectures
def test_notebook_segnet_matches_the_documented_parameter_count():
    module = NotebookSegNet()
    assert sum(p.numel() for p in module.parameters()) == 151_105
    assert module.enc[0].weight.shape == (64, 4, 3, 3)
    module.eval()
    with torch.no_grad():
        output = module(torch.zeros(1, 4, 240, 240))
    assert output.shape == (1, 1, 240, 240)
    assert module.output_semantics == "logit"


def test_notebook_raunet_matches_the_parameter_count_the_notebook_printed():
    module = NotebookRAUNet()
    assert sum(p.numel() for p in module.parameters()) == 1_994_794
    module.eval()
    with torch.no_grad():
        output = module(torch.zeros(1, 4, 240, 240))
    assert output.shape == (1, 1, 240, 240)
    assert module.output_semantics == "logit"


def test_notebook_radiomics_cnn_is_46_wide_and_two_way():
    module = NotebookRadiomicsCNN(46)
    first_linear = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear))
    assert first_linear.in_features == 46
    module.eval()
    with torch.no_grad():
        logits = module(torch.zeros(2, 46))
    assert logits.shape == (2, 2)
    assert module.output_semantics == "class_logits"


def test_the_root_segnet_checkpoint_is_the_same_architecture_with_other_attribute_names():
    """The company file and the notebook file are the same network, differently named.

    ``model.py`` calls the stages ``encoder``/``decoder``; the notebook calls them ``enc``/``dec``.
    Verifying shape-for-shape here means a reader can tell the two apart without loading them,
    and the ensemble never silently substitutes one for the other.
    """

    root = Path(__file__).resolve().parent.parent / "segnet_model.pth"
    if not root.is_file():
        pytest.skip("the company SegNet checkpoint is not present in this checkout")
    state = torch.load(str(root), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    notebook_state = NotebookSegNet().state_dict()
    assert set(state) != set(notebook_state)  # different attribute names
    shapes_by_order = sorted(tuple(value.shape) for value in state.values())
    assert shapes_by_order == sorted(tuple(value.shape) for value in notebook_state.values())


# ------------------------------------------------------------------------------ the registry
def test_the_registry_registers_the_notebook_models_with_their_real_contracts(notebook_config):
    registry = ModelRegistry(notebook_config)
    descriptors = registry.descriptors
    for model_id in NOTEBOOK_MEMBER_FILES:
        descriptor = descriptors[model_id]
        assert descriptor.task == "segmentation"
        assert descriptor.output_semantics == "logit"
        assert descriptor.input_spec.channels == 4
        assert descriptor.input_spec.height == descriptor.input_spec.width == 240
        assert descriptor.input_spec.normalisation_kind == "per_channel"
    radiomics = descriptors["radiomics_ensemble_cnn"]
    assert radiomics.task == "classification"
    assert radiomics.input_spec.channels == 46
    assert radiomics.output_semantics == "class_logits"


def test_every_notebook_checkpoint_loads_and_runs(notebook_config):
    registry = ModelRegistry(notebook_config)
    if not monai_available():
        pytest.skip("install monai to exercise the notebook's U-Net member")
    for model_id in NOTEBOOK_MEMBER_FILES:
        loaded = registry.load_model(model_id)
        assert loaded.is_ready, f"{model_id}: {loaded.status} - {loaded.message}"
        assert loaded.parameter_count and loaded.parameter_count > 0
        assert getattr(loaded.module, "output_semantics", None) == "logit"


def test_a_monai_checkpoint_without_monai_is_reported_as_a_missing_dependency(notebook_config, monkeypatch):
    """Not 'incompatible': the file was never inspected, so the environment is what is wrong."""

    import app.ml.registry as registry_module

    monkeypatch.setattr(
        registry_module.importlib.util, "find_spec", lambda name: None if name == "monai" else object()
    )
    registry = ModelRegistry(notebook_config, )
    loaded = registry.load_model("unet_ensemble_240")
    if not (Path(notebook_config.MODEL_PATHS["unet_ensemble_240"]).is_file()):
        pytest.skip("no U-Net checkpoint in this fixture")
    assert loaded.status == "dependency_missing"
    assert "monai" in loaded.message
    assert loaded.module is None


def test_a_partial_model_paths_mapping_registers_the_ensemble_as_missing_not_as_a_crash():
    """Some deployments (and the Kaggle-notebook test) supply only the original four paths.

    The registry must still build every descriptor and report the ensemble's files as absent,
    because "not installed" is a state the interface is expected to render.
    """

    from dataclasses import dataclass

    from app.config import DevelopmentConfig

    @dataclass
    class Partial(DevelopmentConfig):
        MODEL_PATHS: object = None
        DEVICE_PREFERENCE: str = "cpu"
        TORCH_THREADS: int = 2
        MODEL_CACHE_ENABLED: bool = False

    registry = ModelRegistry(
        Partial(
            MODEL_PATHS={
                "segnet_brats_4ch": "/nowhere/segnet_model.pth",
                "unet_dynamic_512": "/nowhere/unet.pt",
                "raunet_brats_4ch": "/nowhere/raunet_brats_4ch.pt",
                "radiomics_mlp_v1": "/nowhere/radiomics_cnn_model.pth",
            }
        )
    )
    for model_id, file_name in (
        ("segnet_ensemble_240", "segnet.pth"),
        ("unet_ensemble_240", "unet_best.pth"),
        ("raunet_ensemble_240", "raunet.pth"),
        ("radiomics_ensemble_cnn", "radiomics_cnn.pth"),
    ):
        loaded = registry.load_model(model_id)
        assert loaded.status == "missing", model_id
        assert loaded.descriptor.path.endswith(file_name)
        assert Path(loaded.descriptor.path).parent.name == "weights"


def test_notebook_models_are_never_mixed_with_the_company_models_in_one_ensemble(notebook_config):
    from app.services.ensemble_service import EnsembleService
    from app.services.segmentation_service import SegmentationService

    registry = ModelRegistry(notebook_config)
    segmentation = SegmentationService(registry, notebook_config)
    ensemble = EnsembleService(registry, notebook_config, segmentation)
    compatibility = ensemble.compatibility(["segnet_brats_4ch", "segnet_ensemble_240"])
    assert compatibility["available"] is False
    assert all(not group["ensemblable"] for group in compatibility["groups"])
    assert any("different inputs" in reason for reason in compatibility["reasons"])


# --------------------------------------------------------------------------- preprocessing
def test_the_two_normalisations_produce_different_tensors():
    rng = np.random.default_rng(1)
    payloads = []
    for index in range(4):
        array = (rng.normal(0.4, 0.1, size=(32, 32)) * (1 + index)).clip(0, 1)
        buffer = io.BytesIO()
        Image.fromarray((array * 255).astype(np.uint8)).save(buffer, format="PNG")
        payloads.append((f"m{index}.png", buffer.getvalue()))
    global_tensor = preprocess_four_channel(payloads, size=64, normalisation=NORMALISATION_GLOBAL)
    channel_tensor = preprocess_four_channel(
        payloads, size=64, normalisation=NORMALISATION_PER_CHANNEL
    )
    assert not np.allclose(
        global_tensor.tensor.numpy(), channel_tensor.tensor.numpy()
    ), "the two normalisations must not be interchangeable"
    # per-channel: every channel ends up standardised on its own
    for channel in range(4):
        plane = channel_tensor.tensor.numpy()[0, channel]
        assert abs(plane.mean()) < 1e-5 and abs(plane.std() - 1.0) < 1e-3
    assert channel_tensor.metadata["normalisation"] == NORMALISATION_PER_CHANNEL
    assert global_tensor.pipeline == "four_channel_64"


# --------------------------------------------------------------------------- the bundle
def test_status_lists_every_artifact_and_member(services_for_notebook):
    _, _, bundle = services_for_notebook
    status = bundle.status()
    assert [entry["key"] for entry in status["artifacts"]] == [
        "radiomics_cnn",
        "radiomics_scaler",
        "radiomics_feature_schema",
        "meta_scaler",
        "meta_learner",
    ]
    assert all(entry["present"] for entry in status["artifacts"])
    assert [member["model_id"] for member in status["members"]] == list(SEGMENTATION_MEMBERS)
    assert status["contracts"]["radiomics_features"] == 46
    assert status["contracts"]["meta_features"] == META_FEATURE_COUNT == 10


def test_meta_feature_names_are_in_the_notebook_order(services_for_notebook):
    _, _, bundle = services_for_notebook
    names = bundle.status()["meta_features"]
    assert names == [
        "segnet_ensemble_240.dice",
        "segnet_ensemble_240.max_prob",
        "segnet_ensemble_240.mean_prob",
        "unet_ensemble_240.dice",
        "unet_ensemble_240.max_prob",
        "unet_ensemble_240.mean_prob",
        "raunet_ensemble_240.dice",
        "raunet_ensemble_240.max_prob",
        "raunet_ensemble_240.mean_prob",
        "radiomics_prediction",
    ]


def test_a_missing_artifact_is_reported_with_its_file_name(notebook_config, notebook_weights):
    registry = ModelRegistry(notebook_config)

    from app.services.segmentation_service import SegmentationService

    segmentation = SegmentationService(registry, notebook_config)
    bundle = EnsembleBundleService(notebook_config, registry, segmentation)

    class MissingMeta(notebook_config):
        META_LEARNER_PATH = str(notebook_weights / "meta_learner_absent.pkl")

    bundle = EnsembleBundleService(MissingMeta, registry, segmentation)
    status = bundle.status()
    assert status["available"] is False
    assert status["state"] == "artifacts_missing"
    assert "meta_learner_absent.pkl" in status["reason"]
    with pytest.raises(ModelUnavailableError) as error:
        bundle.predict(notebook_tensor())
    assert "meta_learner_absent.pkl" in error.value.message
    # every location that was searched is named, so a reader can see where the file belongs
    searched = error.value.details["searched"]
    assert len(searched) == 5
    assert any(path.endswith("meta_learner_absent.pkl") for path in searched)


def test_a_wrong_feature_schema_is_refused(notebook_config, notebook_weights, tmp_path):
    """A schema that is not the notebook's 46 names must stop the pipeline, not be reordered."""

    schema = tmp_path / "radiomics_feature_names.csv"
    schema.write_text("alpha,beta,gamma\n1,2,3\n")

    from app.services.segmentation_service import SegmentationService

    class WrongSchema(notebook_config):
        RADIOMICS_ENSEMBLE_SCHEMA_PATH = str(schema)

    registry = ModelRegistry(WrongSchema)
    bundle = EnsembleBundleService(WrongSchema, registry, SegmentationService(registry, WrongSchema))
    status = bundle.status()
    assert status["available"] is False
    assert status["state"] == "incompatible"
    assert "does not describe the feature set" in status["reason"]


def test_a_scaler_with_the_wrong_width_is_refused(notebook_config, notebook_weights, tmp_path):
    import joblib
    from sklearn.preprocessing import StandardScaler

    wrong = tmp_path / "radiomics_scaler.pkl"
    joblib.dump(StandardScaler().fit(np.zeros((4, 45))), wrong)

    from app.services.segmentation_service import SegmentationService

    class WrongScaler(notebook_config):
        RADIOMICS_ENSEMBLE_SCALER_PATH = str(wrong)

    registry = ModelRegistry(WrongScaler)
    bundle = EnsembleBundleService(WrongScaler, registry, SegmentationService(registry, WrongScaler))
    status = bundle.status()
    assert status["available"] is False
    assert "expects 45 features" in status["reason"]


def test_a_meta_learner_with_the_wrong_width_is_refused(notebook_config, tmp_path):
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 9))
    learner = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(matrix, (matrix[:, 0] > 0))
    path = tmp_path / "meta_learner_9.pkl"
    joblib.dump(learner, path)

    from app.services.segmentation_service import SegmentationService

    class WrongMeta(notebook_config):
        META_LEARNER_PATH = str(path)

    registry = ModelRegistry(WrongMeta)
    bundle = EnsembleBundleService(WrongMeta, registry, SegmentationService(registry, WrongMeta))
    status = bundle.status()
    assert status["available"] is False
    assert "meta features" in status["reason"] and "9" in status["reason"]
    assert "meta_scaler" in json.dumps(status["contracts"]) or "problems" in status["contracts"]


def test_a_single_channel_tensor_is_refused(services_for_notebook):
    _, _, bundle = services_for_notebook
    with pytest.raises(ValidationError) as error:
        bundle.predict(torch.zeros(1, 1, 240, 240))
    assert "four-channel" in error.value.message


def test_a_tensor_of_the_wrong_spatial_size_is_refused(services_for_notebook):
    _, _, bundle = services_for_notebook
    with pytest.raises(ValidationError) as error:
        bundle.predict(torch.zeros(1, 4, 256, 256))
    assert "240x240" in error.value.message


@pytest.mark.skipif(not monai_available(), reason="install monai to run the notebook's U-Net member")
def test_the_full_pipeline_returns_the_notebooks_ten_meta_features(services_for_notebook):
    _, segmentation, bundle = services_for_notebook
    tensor = notebook_tensor()
    members = {
        model_id: segmentation.predict(model_id, _TensorWrapper(tensor), threshold=0.5)
        for model_id in SEGMENTATION_MEMBERS
    }
    result = bundle.predict(tensor, threshold=0.5, member_results=members)

    assert result["meta_feature_count"] == 10
    assert list(result["meta_features"]) == bundle.status()["meta_features"]
    assert len(result["meta_features_scaled"]) == 10
    assert result["meta_classes"] == ["0", "1"]
    assert result["prediction"] in (0, 1)
    if result["class_probabilities"] is not None:
        assert sum(result["class_probabilities"]) == pytest.approx(1.0, abs=1e-6)
    # the radiomics branch
    assert result["radiomics"]["feature_count"] == 46
    assert len(result["radiomics"]["probabilities"]) == 2
    assert result["radiomics"]["prediction"] in (0, 1)
    # the members really were the ones that produced the statistics
    assert all(member["reused_from_analysis"] for member in result["member_statistics"])
    for entry, model_id in zip(result["member_statistics"], SEGMENTATION_MEMBERS):
        probability = members[model_id].probability
        assert entry["max_probability"] == pytest.approx(float(probability.max()), abs=1e-6)
        assert entry["mean_probability"] == pytest.approx(float(probability.mean()), abs=1e-6)
    # the honest labels
    assert result["dice_reference"] == "ensemble consensus"
    assert "predicted masks" in result["feature_mask_source"] or "consensus" in result["feature_mask_source"]
    assert "proxy" in result["target_semantics"]
    assert result["class_labels"] == ["small_lesion_proxy", "large_lesion_proxy"]
    assert all("HGG" not in label and "grade" not in label.lower() for label in result["class_labels"])


@pytest.mark.skipif(not monai_available(), reason="install monai to run the notebook's U-Net member")
def test_the_prediction_is_deterministic(services_for_notebook):
    _, segmentation, bundle = services_for_notebook
    tensor = notebook_tensor(seed=2)
    members = {
        model_id: segmentation.predict(model_id, _TensorWrapper(tensor), threshold=0.5)
        for model_id in SEGMENTATION_MEMBERS
    }
    first = bundle.predict(tensor, threshold=0.5, member_results=members)
    second = bundle.predict(tensor, threshold=0.5, member_results=members)
    assert first["prediction"] == second["prediction"]
    assert first["meta_features"] == second["meta_features"]
    assert first["radiomics"]["raw_feature_digest"] == second["radiomics"]["raw_feature_digest"]
    assert first["radiomics"]["scaled_feature_digest"] == second["radiomics"]["scaled_feature_digest"]


@pytest.mark.skipif(not monai_available(), reason="install monai to run the notebook's U-Net member")
def test_members_can_be_run_by_the_service_when_none_are_supplied(services_for_notebook):
    _, _, bundle = services_for_notebook
    result = bundle.predict(notebook_tensor(seed=4), threshold=0.5)
    assert all(not member["reused_from_analysis"] for member in result["member_statistics"])
    assert result["meta_feature_count"] == 10


@pytest.mark.skipif(not monai_available(), reason="install monai to run the notebook's U-Net member")
def test_a_numpy_slice_is_accepted_when_the_members_have_to_be_run_here(services_for_notebook):
    """``scripts/audit_integration.py`` drives ``predict`` with a NumPy array.

    The segmentation service moves ``.tensor`` onto the device itself, so the wrapper has to
    hand it a tensor rather than the raw array; this locks that behaviour in.
    """

    _, _, bundle = services_for_notebook
    tensor = np.asarray(notebook_tensor(seed=5))
    assert isinstance(tensor, np.ndarray)
    result = bundle.predict(tensor, threshold=0.5)
    assert result["available"] is True
    assert result["meta_feature_count"] == 10
    assert all(not member["reused_from_analysis"] for member in result["member_statistics"])


def test_an_unavailable_member_stops_the_pipeline_instead_of_being_zero_filled(notebook_config):
    """The meta-learner was fitted on three members; two is a different matrix."""

    from app.services.segmentation_service import SegmentationService

    class NoRaunet(notebook_config):
        MODEL_PATHS = {**notebook_config.MODEL_PATHS, "raunet_ensemble_240": "/nowhere/raunet.pth"}

    registry = ModelRegistry(NoRaunet)
    bundle = EnsembleBundleService(NoRaunet, registry, SegmentationService(registry, NoRaunet))
    status = bundle.status()
    assert status["available"] is False
    assert status["state"] == "members_missing"
    assert "raunet_ensemble_240" in status["reason"]


def test_reported_metrics_are_labelled_as_reported(services_for_notebook):
    _, _, bundle = services_for_notebook
    status = bundle.status()
    assert status["reported_metrics"]["stacked_ensemble_test_accuracy"] == pytest.approx(0.9633)
    for sentence in (
        "have not been re-measured",
        "not shipped with the weights",
    ):
        assert sentence.lower() in status["metrics_provenance"].lower()
    note = status["leakage_note"].lower()
    assert "target leakage" in note and "split leakage" in note
    assert "grade" in status["target_semantics"].lower()


def test_the_status_says_it_is_not_the_grade_or_type_model(services_for_notebook):
    _, _, bundle = services_for_notebook
    text = bundle.status()["distinct_from_grade_and_type_models"]
    assert "HGG" in text and "Xception" in text and "never" in text


# ------------------------------------------------------------------------------- the API
@pytest.fixture()
def notebook_client(notebook_config, tmp_path):
    from app import create_app

    class ApiConfig(notebook_config):
        DATA_DIR = tmp_path / "api-data"
        LOG_DIR = tmp_path / "api-data" / "logs"

    ApiConfig.init_directories()
    application = create_app(config_object=ApiConfig, env="testing")
    application.config.update(TESTING=True)
    return application.test_client()


def test_the_stacked_status_endpoint_reports_every_artifact(notebook_client):
    response = notebook_client.get("/api/ensemble/stacked")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task"] == "stacked_ensemble"
    assert payload["input_requirements"]["spatial_size"] == [240, 240]
    assert payload["input_requirements"]["analysis_type"] == "four_channel"
    assert len(payload["artifacts"]) == 5
    assert payload["target_semantics"]


def test_the_models_endpoint_lists_the_notebook_checkpoints(notebook_client):
    ids = {entry["model_id"] for entry in notebook_client.get("/api/models").get_json()["models"]}
    assert set(SEGMENTATION_MEMBERS) | {"radiomics_ensemble_cnn"} <= ids


def test_capabilities_carry_the_stacked_summary(notebook_client):
    summary = notebook_client.get("/api/capabilities").get_json()["stacked_ensemble"]
    assert summary is not None
    assert summary["state"] in {"ready", "artifacts_missing", "members_missing", "incompatible"}
    assert isinstance(summary["missing_artifacts"], list)
    assert summary["reported_metrics"]["stacked_ensemble_validation_accuracy"] == pytest.approx(0.9733)
    assert "grade" in summary["target_semantics"].lower()


def test_the_run_endpoint_needs_an_analysis_id(notebook_client):
    response = notebook_client.post("/api/ensemble/stacked/run", json={})
    assert response.status_code == 422
    assert response.get_json()["error"]["details"]["field"] == "analysis_id"


def test_the_run_endpoint_refuses_a_missing_analysis(notebook_client):
    response = notebook_client.post(
        "/api/ensemble/stacked/run", json={"analysis_id": "an-20260101-0000000000"}
    )
    assert response.status_code == 404


@pytest.mark.skipif(not monai_available(), reason="install monai to run the notebook's U-Net member")
def test_the_stacked_ensemble_runs_through_the_analysis_api(notebook_client):
    """The whole HTTP path: four modality uploads -> members -> meta-learner -> payload."""

    rng = np.random.default_rng(9)
    size = 240
    grid = np.mgrid[0:size, 0:size]
    lesion = np.exp(-(((grid[0] - size / 2) ** 2 + (grid[1] - size / 2) ** 2) / (2 * (size * 0.08) ** 2)))
    data = {
        "analysis_type": "four_channel",
        "enable_stacked_ensemble": "true",
        "enable_ensemble": "true",
        "enable_radiomics": "true",
        "enable_explainability": "false",
    }
    from werkzeug.datastructures import MultiDict

    form = MultiDict(list(data.items()))
    for index, gain in enumerate([1.0, 1.1, 0.9, 1.25]):
        array = np.clip(rng.normal(0.35, 0.08, size=(size, size)) + lesion * 0.6 * gain, 0, 1)
        buffer = io.BytesIO()
        Image.fromarray((array * 255).astype(np.uint8)).save(buffer, format="PNG")
        form.add("files", (io.BytesIO(buffer.getvalue()), f"modality_{index}.png"))

    response = notebook_client.post("/api/analysis", data=form, content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    payload = response.get_json()

    for model_id in SEGMENTATION_MEMBERS:
        entry = payload["models"].get(model_id)
        assert entry is not None and entry["available"], f"{model_id} did not run"
        assert entry["summary"]["input_shape"] == [1, 4, 240, 240]

    stacked = payload["stacked_ensemble"]
    assert stacked["available"] is True, stacked.get("reasons")
    assert stacked["meta_feature_count"] == 10
    assert stacked["radiomics"]["feature_count"] == 46
    assert stacked["consensus"]["foreground_pixels"] > 0
    assert stacked["prediction"] in (0, 1)
    assert stacked["class_probabilities"] is None or len(stacked["class_probabilities"]) == 2
    assert stacked["dice_reference"] == "ensemble consensus"
    assert "proxy" in stacked["target_semantics"]

    # the rest of the platform still worked in the same request
    assert payload["radiomics"]["available"] is True
    assert payload["region_measurements"]["detected"] is True
    assert payload["analysis_id"].startswith("an-")


def test_the_run_endpoint_rescores_a_stored_analysis(notebook_client):
    """``POST /api/ensemble/stacked/run`` executes the members and the head again.

    The result must agree with the one the analysis recorded (same inputs, same weights), and the
    stored payload must be left untouched.
    """

    from werkzeug.datastructures import MultiDict

    rng = np.random.default_rng(11)
    size = 240
    grid = np.mgrid[0:size, 0:size]
    lesion = np.exp(-(((grid[0] - size / 2) ** 2 + (grid[1] - size / 2) ** 2) / (2 * (size * 0.08) ** 2)))
    form = MultiDict(
        [
            ("analysis_type", "four_channel"),
            ("enable_stacked_ensemble", "true"),
            ("enable_radiomics", "false"),
            ("enable_ensemble", "false"),
            ("enable_explainability", "false"),
        ]
    )
    for index, gain in enumerate([1.0, 1.15, 0.85, 1.3]):
        array = np.clip(rng.normal(0.35, 0.08, size=(size, size)) + lesion * 0.6 * gain, 0, 1)
        buffer = io.BytesIO()
        Image.fromarray((array * 255).astype(np.uint8)).save(buffer, format="PNG")
        form.add("files", (io.BytesIO(buffer.getvalue()), f"rerun_{index}.png"))

    created = notebook_client.post("/api/analysis", data=form, content_type="multipart/form-data")
    assert created.status_code == 201, created.get_json()
    payload = created.get_json()
    analysis_id = payload["analysis_id"]
    stored = payload["stacked_ensemble"]
    assert stored["available"] is True, stored.get("reasons")

    response = notebook_client.post(
        "/api/ensemble/stacked/run", json={"analysis_id": analysis_id}
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["persisted"] is False
    rerun = body["stacked_ensemble"]
    assert rerun["available"] is True, rerun.get("reasons")

    # same input, same weights -> the same numbers, and all three members were executed for real
    assert rerun["prediction"] == stored["prediction"]
    assert rerun["meta_features"] == stored["meta_features"]
    assert rerun["consensus"]["foreground_pixels"] == stored["consensus"]["foreground_pixels"]
    assert rerun["member_source"] == (
        "the three members were executed for this call from the analysis' stored uploads"
    )
    assert rerun["source_analysis"] == analysis_id
    assert rerun["uploads_used"] == [f"rerun_{index}.png" for index in range(4)]

    # the stored payload still holds the original result
    again = notebook_client.get(f"/api/analysis/{analysis_id}").get_json()["result"]
    assert again["stacked_ensemble"]["meta_features"] == stored["meta_features"]


def test_the_run_endpoint_refuses_a_single_channel_analysis(notebook_client):
    buffer = io.BytesIO()
    rng = np.random.default_rng(12)
    Image.fromarray((rng.random((256, 256)) * 255).astype(np.uint8)).save(buffer, format="PNG")
    created = notebook_client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "files": (io.BytesIO(buffer.getvalue()), "one.png"),
        },
        content_type="multipart/form-data",
    )
    assert created.status_code == 201, created.get_json()
    analysis_id = created.get_json()["analysis_id"]
    response = notebook_client.post("/api/ensemble/stacked/run", json={"analysis_id": analysis_id})
    assert response.status_code == 422
    error = response.get_json()["error"]
    assert "single_channel" in error["message"]


def test_the_analysis_api_refuses_the_stacked_ensemble_on_a_single_channel_upload(notebook_client):
    rng = np.random.default_rng(4)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((256, 256)) * 255).astype(np.uint8)).save(buffer, format="PNG")
    response = notebook_client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "enable_stacked_ensemble": "true",
            "files": (io.BytesIO(buffer.getvalue()), "slice.png"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    message = response.get_json()["error"]["message"]
    assert "four-channel" in message
