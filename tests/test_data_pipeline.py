"""Tests for the cached-array data pipeline the notebook trains from.

The pipeline lives in `notebooks/_notebook_source/nbs_02_data.py` (cache, index, payload
store) and `nbs_03_splits_trainer.py` (batch iteration, benchmark). The functions are exec'd
out of those files, so what is tested here is the shipped code, not a copy of it.

What is pinned down:

* **The optimisation must change nothing observable.** Batches from the new reader are
  compared bit for bit with `iterate_batches_legacy`, the implementation it replaced: same
  images, same masks, same order, same channel order (T1, T1ce, T2, FLAIR), same dtypes.
* **The root cause must not come back.** The old loop inflated a patient's entire
  compressed archive once per patient (to count slices) and once per slice (to build a
  batch). These tests count member inflations directly, so a future change that starts
  reading a whole archive per slice fails here instead of silently costing hours on Kaggle.
* **Memory stays bounded** (byte budget and open-memory-map cap) and a cache conversion is
  verified per member and must never delete an archive it could not convert.

Every array here is random test data. No number in this file is a result.
"""
from __future__ import annotations

import ast
import contextlib
import json
import queue
import shutil
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")          # the batching helpers resize masks through torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "notebooks" / "_notebook_source"
CACHE_SOURCE = SOURCE / "nbs_02_data.py"
TRAINER_SOURCE = SOURCE / "nbs_03_splits_trainer.py"

CACHE_FUNCTIONS = {
    "resize2d",
    "zscore_channels",
    "resize2d_channels",
    "single_pair",
    "CacheError",
    "cache_archive",
    "cache_directory",
    "cache_member_path",
    "cache_meta_path",
    "cache_layout",
    "cache_entry_path",
    "cached_volumes",
    "round_mib",
    "layout_bytes",
    "read_npy_header",
    "read_archive_member_header",
    "probe_patient",
    "load_cache_index",
    "write_cache_index",
    "cache_index_entry",
    "patient_slice_count",
    "write_npy_part",
    "parts_of",
    "clean_parts",
    "write_cache_members",
    "build_patient_cache",
    "upgrade_patient_cache",
    "PatientPayloadStore",
    "patient_store",
    "load_patient",
    "measure_cache_io",
    "validate_cache",
}
CACHE_CONSTANTS = {
    "CACHE_MEMBERS",
    "MEMORY_MAP_FORMAT",
    "ARCHIVE_FORMAT",
    "CACHE_INDEX_NAME",
    "CACHE_DIRECTORY_SUFFIX",
    "CACHE_INDEX",
    "CACHE_INDEX_PATH",
    "CACHE_INDEX_PROBES",
    "PATIENT_STORE",
}
BATCH_FUNCTIONS = {
    "store_activity",
    "upsample_mask_nearest",
    "iterate_batches_legacy",
    "prepare_batch",
    "iterate_batches",
    "prefetched_batches",
    "summarise_seconds",
    "benchmark_data_pipeline",
}


def _names_of(node: ast.stmt) -> set[str]:
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, ast.Assign):
        names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.add(node.target.id)
    return names


def definitions(path: Path, wanted: set[str]) -> list[ast.stmt]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if _names_of(node) & wanted]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return module


def make_namespace(cache_dir: Path, *, run: dict | None = None, read_slice=None, recorder=None) -> dict:
    """A namespace holding the notebook's real cache and batching code, plus test stubs."""
    from PIL import Image as PILImage

    namespace: dict = {
        "np": np,
        "Path": Path,
        "time": time,
        "json": json,
        "shutil": shutil,
        "zipfile": zipfile,
        "OrderedDict": OrderedDict,
        "queue": queue,
        "threading": threading,
        "contextlib": contextlib,
        "PILImage": PILImage,
        "SEED": 20240101,
        "PATHS": {"cache": Path(cache_dir), "artifacts": Path(cache_dir) / "_artifacts", "work": Path(cache_dir) / "_work"},
        "RUN": {
            "cache_format": "memmap",
            "upgrade_cache_to_memmap": True,
            "patient_cache_mb": 256,
            "patient_cache_handles": 8,
            "batch_size": 8,
            "image_size": 32,
            "single_image_size": 64,
            "single_channel_index": 3,
            "prefetch_batches": False,
            "benchmark_steps": 2,
            "benchmark_warmup": 1,
        },
        "write_json": lambda path, payload: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8"),
        )[-1],
        "read_json": lambda path, default=None: (
            json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
        ),
        "utc_now": lambda: "1970-01-01T00:00:00+00:00",
        "read_slice": read_slice or _unsupported_read_slice,
    }
    namespace["PATHS"]["artifacts"].mkdir(parents=True, exist_ok=True)
    if run:
        namespace["RUN"].update(run)
    torch = pytest.importorskip("torch")
    namespace["torch"] = torch
    real_numpy, real_zipfile = namespace["np"], namespace["zipfile"]
    if recorder is not None:
        namespace["np"] = CountingNumpy(real_numpy, recorder)
        namespace["zipfile"] = CountingZip(real_zipfile, recorder)
    # The exec'd functions resolve `np`/`zipfile` from this namespace at call time, so the
    # counting stand-ins stay in place: they forward everything and only record.
    exec(compile(definitions(CACHE_SOURCE, CACHE_FUNCTIONS | CACHE_CONSTANTS), str(CACHE_SOURCE), "exec"), namespace)
    exec(compile(definitions(TRAINER_SOURCE, BATCH_FUNCTIONS), str(TRAINER_SOURCE), "exec"), namespace)
    namespace["_real_numpy"], namespace["_real_zipfile"] = real_numpy, real_zipfile
    return namespace


def _unsupported_read_slice(path):  # pragma: no cover - only called if a test builds a cache without a stub
    raise AssertionError(f"read_slice({path}) was called but this test did not stub it")


class ReadRecorder:
    """Counts the operations the bottleneck was made of.

    `np_loads`           every `np.load(...)` call (an archive open or a memory map)
    `member_inflations`  a whole member read out of an archive -- the deflate work itself
    `mmap_opens`         a memory-mapped member opened
    `header_reads`       a `.npy` header read out of an archive (a few hundred bytes)
    """

    def __init__(self):
        self.np_loads = 0
        self.member_inflations = 0
        self.mmap_opens = 0
        self.header_reads = 0

    def report(self) -> dict:
        return {
            "np_loads": self.np_loads,
            "member_inflations": self.member_inflations,
            "mmap_opens": self.mmap_opens,
            "header_reads": self.header_reads,
        }

    def reset(self):
        self.np_loads = 0
        self.member_inflations = 0
        self.mmap_opens = 0
        self.header_reads = 0


class _TrackedArchive:
    """An open NpzFile whose member reads are counted."""

    def __init__(self, inner, recorder):
        self._inner = inner
        self._recorder = recorder

    @property
    def files(self):
        return self._inner.files

    def __getitem__(self, key):
        self._recorder.member_inflations += 1
        return self._inner[key]

    def __contains__(self, key):
        return key in self._inner.files

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)


class CountingNumpy:
    """A `numpy` stand-in for the notebook code, so counting never touches the real module.

    Only `load` is instrumented; everything else is forwarded, including `np.lib.format`,
    `np.memmap` and `isinstance` checks against it.
    """

    def __init__(self, module, recorder, *, break_member=None):
        self._module = module
        self._recorder = recorder
        self._break_member = break_member

    def __getattr__(self, name):
        return getattr(self._module, name)

    def load(self, path, *args, **kwargs):
        self._recorder.np_loads += 1
        mmap_mode = kwargs.get("mmap_mode")
        is_archive = str(path).endswith(".npz")
        if is_archive and mmap_mode in (None, False):
            inner = self._module.load(path, *args, **kwargs)
            if self._break_member and str(path).endswith(self._break_member):
                return _Corrupt(_TrackedArchive(inner, self._recorder))
            return _TrackedArchive(inner, self._recorder)
        if mmap_mode not in (None, False):
            self._recorder.mmap_opens += 1
            if self._break_member and str(path).endswith(self._break_member):
                return _Corrupt(self._module.load(path, *args, **kwargs))
        return self._module.load(path, *args, **kwargs)


class _Corrupt:
    """Returns slightly wrong data, standing in for a write that did not survive."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def shape(self):
        shape = list(self._inner.shape)
        shape[0] = shape[0] + 1
        return tuple(shape)

    @property
    def dtype(self):
        return self._inner.dtype

    def __getitem__(self, key):
        return self._inner[key]

    def __array__(self, *args, **kwargs):
        return self._inner.__array__(*args, **kwargs)


class CountingZip:
    """`zipfile` stand-in: counts header reads (the only thing the archive index does)."""

    def __init__(self, module, recorder):
        self._module = module
        self._recorder = recorder
        self.ZipFile = self._zip_file()

    def __getattr__(self, name):
        return getattr(self._module, name)

    def _zip_file(self):
        recorder = self._recorder
        real = self._module.ZipFile

        class ZipFile(real):
            def open(self, *args, **kwargs):
                recorder.header_reads += 1
                return super().open(*args, **kwargs)

        return ZipFile


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
SLICES = 6
HEIGHT = WIDTH = 48          # a real BraTS slice is 240x240; the shapes are what matters here
SINGLE = 64


def fake_read_slice(path):
    """A deterministic stand-in for the HDF5 reader.

    Channel c is filled with (c + 1) so the channel order can be checked after the whole
    cache -> store -> batching path, and the mask is a block in the middle of the image.
    """
    index = int(str(path).rsplit("/", 1)[-1])
    image = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    for channel in range(4):
        image[..., channel] = float(channel + 1) + 0.01 * index
    image[4:12, 5:15, :] += 100.0                       # a "lesion" in every channel
    mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    mask[5:11, 6:14, 0] = 1
    mask[6:10, 7:13, 2] = 1                             # sub-regions; only the union matters
    return image, (mask > 0).any(axis=-1)


def write_archive(path: Path, *, slices=SLICES, grade="HGG", seed=0, dtype=np.float16) -> Path:
    """One patient in the legacy compressed layout, with the members a real cache has."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=rng.standard_normal((slices, HEIGHT, WIDTH, 4)).astype(dtype),
        masks=(rng.random((slices, HEIGHT, WIDTH)) < 0.2).astype(np.uint8),
        singles=rng.random((slices, SINGLE, SINGLE)).astype(dtype),
        grade=np.array([grade]),
        slices=np.array([f"/data/{path.stem}/slice_{index}" for index in range(slices)]),
    )
    return path


@pytest.fixture
def archive_cache(tmp_path):
    cache = tmp_path / "cache"
    volumes = [f"patient_{index:03d}" for index in range(4)]
    for index, volume in enumerate(volumes):
        write_archive(cache / f"{volume}.npz", seed=index)
    return cache, volumes


@pytest.fixture
def memmap_cache(tmp_path):
    cache = tmp_path / "cache"
    volumes = [f"patient_{index:03d}" for index in range(3)]
    for index, volume in enumerate(volumes):
        write_archive(cache / f"{volume}.npz", seed=index)
    namespace = make_namespace(cache)
    report = namespace["upgrade_patient_cache"]()
    assert len(report["converted"]) == len(volumes)
    return cache, volumes


# ---------------------------------------------------------------------------
# the cache itself
# ---------------------------------------------------------------------------
def test_build_writes_the_memory_mappable_layout(tmp_path):
    cache = tmp_path / "cache"
    volumes = {"patient_000": {"slices": [f"/data/p0/{index}" for index in range(SLICES)], "grade": "HGG"}}
    namespace = make_namespace(cache, read_slice=fake_read_slice, run={"image_size": 32, "single_image_size": 64})

    report = namespace["build_patient_cache"]("patient_000", volumes["patient_000"], 32, 64, 3)
    assert report["layout"] == "npy" and report["cached"] is False
    assert namespace["cache_layout"]("patient_000") == "npy"
    assert not namespace["cache_archive"]("patient_000").exists(), "the default layout is not the archive"

    images = np.load(namespace["cache_member_path"]("patient_000", "images"))
    masks = np.load(namespace["cache_member_path"]("patient_000", "masks"))
    singles = np.load(namespace["cache_member_path"]("patient_000", "singles"))
    assert images.shape == (SLICES, 32, 32, 4) and images.dtype == np.float16
    assert masks.shape == (SLICES, 32, 32) and masks.dtype == np.uint8
    assert singles.shape == (SLICES, 64, 64) and singles.dtype == np.float16
    assert namespace["patient_slice_count"]("patient_000") == SLICES


def test_channel_order_normalisation_and_masks_are_preserved(tmp_path):
    """The cache contract: z-scored channels in T1/T1ce/T2/FLAIR order, binary union masks."""
    cache = tmp_path / "cache"
    entry = {"slices": [f"/data/p0/{index}" for index in range(SLICES)], "grade": "HGG"}
    namespace = make_namespace(cache, read_slice=fake_read_slice)
    namespace["build_patient_cache"]("p0", entry, 32, 64, 3)

    payload = namespace["load_patient"]("p0")
    images, masks, singles = payload["images"], payload["masks"], payload["singles"]

    # normalisation: one global z-score over the (H, W, C) block
    block = images[0].astype(np.float32)
    assert abs(float(block.mean())) < 1e-3
    assert abs(float(block.std()) - 1.0) < 1e-3

    # channel order: channel c came from channel c of the source slice, in order
    channel_means = [float(images[0][..., channel].astype(np.float32).mean()) for channel in range(4)]
    assert channel_means == sorted(channel_means), "channel order must be preserved (T1, T1ce, T2, FLAIR)"
    assert len(set(round(value, 6) for value in channel_means)) == 4

    # masks: the union of the stored sub-region planes, nearest-resized, still binary
    image, whole = fake_read_slice("/data/p0/0")
    from PIL import Image as PILImage

    expected = np.asarray(
        PILImage.fromarray(whole.astype(np.float32), mode="F").resize((32, 32), resample=PILImage.NEAREST)
    ) > 0.5
    assert masks[0].max() <= 1, "the whole-tumour mask must stay binary"
    assert np.array_equal(masks[0].astype(bool), expected)
    assert payload["grade"] == "HGG"
    assert len(payload["slices"]) == SLICES

    # single-channel input: the FLAIR channel (index 3), scaled with the / 255 contract
    flair = image[..., 3]
    resized = np.asarray(PILImage.fromarray(flair.astype(np.float32), mode="F").resize((64, 64), resample=PILImage.BILINEAR))
    assert singles.shape == (SLICES, 64, 64)
    assert np.allclose(singles[0].astype(np.float32), resized / 255.0, atol=1e-3)
    assert float(singles.min()) >= 0.0 and float(singles.max()) <= 1.0


def test_slice_index_reads_headers_only(archive_cache):
    """Counting slices used to cost one whole payload load per patient."""
    cache, volumes = archive_cache
    recorder = ReadRecorder()
    namespace = make_namespace(cache, recorder=recorder)
    namespace["load_cache_index"](rebuild=True)
    recorder.reset()
    for volume in volumes:
        assert namespace["patient_slice_count"](volume) == SLICES
        assert namespace["probe_patient"](volume)["members"]["images"] == [SLICES, HEIGHT, WIDTH, 4]
    report = recorder.report()
    assert report["member_inflations"] == 0, f"the index inflated archive members: {report}"
    assert report["np_loads"] == 0, f"the index loaded payloads: {report}"


def test_the_old_reader_pays_one_archive_per_slice(archive_cache):
    """The measured root cause, pinned: reading slice after slice used to re-inflate the archive.

    `load_patient(..., legacy_archive=True)` is the read the old batch loop performed for every
    single slice; the store performs one member-selective read per patient instead.
    """
    cache, volumes = archive_cache
    recorder = ReadRecorder()
    namespace = make_namespace(cache, recorder=recorder)
    volume = volumes[0]

    namespace["patient_store"](reset=True)
    recorder.reset()
    for _ in range(SLICES):
        namespace["load_patient"](volume, legacy_archive=True)
    old = recorder.report()

    namespace["patient_store"](reset=True)
    recorder.reset()
    for _ in range(SLICES):
        namespace["load_patient"](volume)
    new = recorder.report()

    assert old["member_inflations"] >= SLICES * 3, f"the old reader should inflate per slice: {old}"
    assert new["member_inflations"] <= 3 + 2, f"the store should read one member set per patient: {new}"
    assert new["member_inflations"] * 3 <= old["member_inflations"], f"{new} vs {old}"


def test_the_batch_reader_reads_per_patient_not_per_slice(archive_cache):
    """Batch iteration: one read per patient in the batch, and no payload loads for the index."""
    cache, volumes = archive_cache
    recorder = ReadRecorder()
    namespace = make_namespace(cache, recorder=recorder)

    namespace["patient_store"](reset=True)
    recorder.reset()
    generator = namespace["iterate_batches_legacy"](
        volumes, batch_size=4, size=32, shuffle=False, rng=np.random.default_rng(0), max_steps=1
    )
    next(generator)
    legacy = recorder.report()

    namespace["patient_store"](reset=True)
    recorder.reset()
    generator = namespace["iterate_batches"](
        volumes, batch_size=4, size=32, shuffle=False, rng=np.random.default_rng(0), max_steps=1
    )
    next(generator)
    optimised = recorder.report()

    # the old generator loads a payload for every patient (index) and for every slice
    assert legacy["member_inflations"] >= 3 * len(volumes), legacy
    # the new one reads headers for the index and at most images + masks per patient in the batch
    assert optimised["member_inflations"] <= 2 * len(volumes), optimised
    assert optimised["member_inflations"] < legacy["member_inflations"]


def test_slices_are_read_from_memory_maps_without_inflating(memmap_cache):
    cache, volumes = memmap_cache
    recorder = ReadRecorder()
    namespace = make_namespace(cache, recorder=recorder)
    namespace["patient_store"](reset=True)
    recorder.reset()
    generator = namespace["iterate_batches"](
        volumes, batch_size=6, size=32, shuffle=True, rng=np.random.default_rng(3), max_steps=3
    )
    batches = list(generator)
    report = recorder.report()
    assert len(batches) == 3
    assert report["member_inflations"] == 0, f"a memory-mapped cache must not inflate anything: {report}"
    assert report["mmap_opens"] <= 2 * len(volumes), report


# ---------------------------------------------------------------------------
# equivalence with the implementation that was replaced
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("single_channel", [False, True])
@pytest.mark.parametrize("seed", [0, 7])
def test_batches_are_identical_to_the_legacy_reader(memmap_cache, single_channel, seed):
    cache, volumes = memmap_cache
    size = 64 if single_channel else 32
    namespace = make_namespace(cache)
    batches = {}
    for label, legacy in (("legacy", True), ("optimised", False)):
        namespace["patient_store"](reset=True)
        batches[label] = list(
            namespace["iterate_batches"](
                volumes,
                batch_size=4,
                size=size,
                single_channel=single_channel,
                shuffle=True,
                rng=np.random.default_rng(seed),
                max_steps=3,
                legacy=legacy,
            )
        )
    assert len(batches["legacy"]) == len(batches["optimised"]) == 3
    for index, (before, after) in enumerate(zip(batches["legacy"], batches["optimised"])):
        assert np.array_equal(before[0], after[0]), f"images differ in batch {index}"
        assert np.array_equal(before[1], after[1]), f"masks differ in batch {index}"
        assert before[0].dtype == after[0].dtype == np.float32
        assert before[1].dtype == after[1].dtype == np.float32


def test_batch_composition_is_unchanged_without_shuffling(memmap_cache):
    """shuffle=False must walk patients and slices in the cached order, as it always did."""
    cache, volumes = memmap_cache
    namespace = make_namespace(cache)
    reference = list(
        namespace["iterate_batches"](volumes, batch_size=6, size=32, shuffle=False, max_steps=2, legacy=True)
    )
    optimised = list(
        namespace["iterate_batches"](volumes, batch_size=6, size=32, shuffle=False, max_steps=2)
    )
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(reference, optimised))
    # the first batch is slices 0..5 of the first patient: contiguous, in order
    payload = namespace["load_patient"](volumes[0])
    for slot in range(6):
        expected = np.moveaxis(payload["images"][slot].astype(np.float32), -1, 0)
        assert np.array_equal(optimised[0][0][slot], expected)


def test_prefetching_produces_the_same_batches_and_leaves_no_thread_behind(memmap_cache):
    cache, volumes = memmap_cache
    namespace = make_namespace(cache)
    baseline_threads = threading.active_count()
    direct = list(
        namespace["iterate_batches"](volumes, batch_size=4, size=32, shuffle=True, rng=np.random.default_rng(1), prefetch=False)
    )
    namespace["patient_store"](reset=True)
    prefetched = list(
        namespace["iterate_batches"](volumes, batch_size=4, size=32, shuffle=True, rng=np.random.default_rng(1), prefetch=True)
    )
    assert len(direct) == len(prefetched)
    for before, after in zip(direct, prefetched):
        assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    deadline = time.time() + 5
    while threading.active_count() > baseline_threads and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= baseline_threads, "the prefetch worker did not shut down"


def test_prefetching_stops_cleanly_when_the_consumer_breaks_early(memmap_cache):
    """A time-boxed epoch breaks out of the loop; the worker must not keep producing."""
    cache, volumes = memmap_cache
    namespace = make_namespace(cache)
    started = time.time()
    iterator = namespace["iterate_batches"](volumes, batch_size=2, size=32, shuffle=True, prefetch=True)
    next(iterator)
    iterator.close()
    assert time.time() - started < 30, "closing a prefetching iterator must not block"


# ---------------------------------------------------------------------------
# bounded memory
# ---------------------------------------------------------------------------
def test_the_store_keeps_memory_bounded(archive_cache):
    cache, volumes = archive_cache
    namespace = make_namespace(cache)
    store = namespace["patient_store"](budget_bytes=200_000, reset=True)
    one_payload = int(np.asarray(namespace["load_patient"](volumes[0])["images"]).nbytes)
    for volume in volumes:
        payload = namespace["load_patient"](volume)
        assert payload["images"].shape[0] == SLICES
    report = store.stats_report()
    assert store.budget_bytes == 200_000
    assert report["materialised_mib"] * 1024 ** 2 <= 200_000 + one_payload, f"the byte budget was exceeded: {report}"
    assert store.stats["evictions"] >= 1, "nothing was evicted, so the budget was never reached"


def test_the_store_keeps_open_memory_maps_bounded(memmap_cache):
    cache, volumes = memmap_cache
    namespace = make_namespace(cache)
    store = namespace["patient_store"](max_handles=1, reset=True)
    for volume in volumes:
        rows = store.slices(volume, [0, 1], member="images")
        assert rows.shape == (2, HEIGHT, WIDTH, 4)
    report = store.stats_report()
    assert report["max_handles"] == 1
    assert report["open_memory_maps"] <= 1, f"the handle cap was exceeded: {report}"
    assert store.stats["member_reads"] >= len(volumes), "every patient should have been opened at least once"


def test_open_file_descriptors_stay_bounded_over_many_patients(tmp_path):
    """A 369-volume run must not run out of file descriptors.

    Memory maps hold descriptors, so the store keeps them in a capped handle table instead of
    one entry per patient. This reads slices from more patients than the cap allows and counts
    the descriptors pointing into the cache directory.
    """
    cache = tmp_path / "cache"
    volumes = [f"patient_{index:03d}" for index in range(12)]
    for index, volume in enumerate(volumes):
        write_archive(cache / f"{volume}.npz", seed=index)
    namespace = make_namespace(cache)
    report = namespace["upgrade_patient_cache"]()
    assert len(report["converted"]) == len(volumes)

    def cache_descriptors():
        count = 0
        for entry in Path("/proc/self/fd").iterdir():
            try:
                target = str(entry.resolve())
            except OSError:  # closed while listing
                continue
            if target.startswith(str(cache)):
                count += 1
        return count

    store = namespace["patient_store"](max_handles=2, reset=True)
    for volume in volumes:
        rows = store.slices(volume, [0, 1, 2], member="images")
        masks = store.slices(volume, [0], member="masks")
        assert rows.shape == (3, HEIGHT, WIDTH, 4) and masks.shape == (1, HEIGHT, WIDTH)
    descriptors = cache_descriptors()
    assert descriptors <= 2 * store.max_handles + 2, (
        f"{descriptors} cache descriptors open for {len(volumes)} patients with max_handles={store.max_handles}"
    )
    assert store.stats["member_reads"] >= len(volumes) - 1


def test_payload_arrays_survive_eviction(archive_cache):
    """A caller keeps its arrays even after the store drops the patient."""
    cache, volumes = archive_cache
    namespace = make_namespace(cache)
    store = namespace["patient_store"](budget_bytes=200_000, reset=True)
    first = namespace["load_patient"](volumes[0])
    check = np.array(first["masks"])
    for volume in volumes[1:]:
        namespace["load_patient"](volume)
    assert store.stats["evictions"] >= 1
    assert np.array_equal(first["masks"], check), "an evicted payload was mutated or freed"


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def test_conversion_is_verified_before_the_archive_is_deleted(archive_cache):
    cache, volumes = archive_cache
    namespace = make_namespace(cache)
    before = {
        volume: {
            member: np.asarray(namespace["np"].load(cache / f"{volume}.npz")[member]) for member in ("images", "masks", "singles")
        }
        for volume in volumes
    }
    report = namespace["upgrade_patient_cache"]()
    assert report["converted"] == volumes and not report["failed"]
    for volume in volumes:
        assert namespace["cache_layout"](volume) == "npy"
        assert not (cache / f"{volume}.npz").exists(), "the archive is only deleted after verification"
        for member, expected in before[volume].items():
            stored = np.load(namespace["cache_member_path"](volume, member))
            assert stored.dtype == expected.dtype and np.array_equal(stored, expected)
        meta = json.loads(namespace["cache_meta_path"](volume).read_text(encoding="utf-8"))
        assert meta["grade"] == "HGG" and len(meta["slices"]) == SLICES


def test_a_failed_conversion_keeps_the_archive_and_reports_it(tmp_path):
    cache = tmp_path / "cache"
    archive = write_archive(cache / "patient_000.npz", seed=1)
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                     # corrupt the middle of the deflate stream
    archive.write_bytes(bytes(raw))

    namespace = make_namespace(cache)
    report = namespace["upgrade_patient_cache"]()
    assert report["failed"], "a corrupt archive must be reported, not silently converted"
    assert archive.exists(), "the archive must survive a failed conversion"
    assert namespace["cache_layout"]("patient_000") == "npz", "a partial conversion must not look complete"
    assert not namespace["cache_member_path"]("patient_000", "images").exists()
    for part in namespace["parts_of"]("patient_000").values():
        assert not part.exists(), f"{part} was left behind"


def test_a_write_that_fails_verification_is_not_published(tmp_path):
    """The publish-last protocol: a bad member can never make the cache look complete."""
    cache = tmp_path / "cache"
    namespace = make_namespace(cache)
    payload = np.zeros((2, 4, 4), dtype=np.uint8)
    # make the read-back of the images member disagree with what was written
    namespace["np"] = CountingNumpy(np, ReadRecorder(), break_member="images.npy.part")
    namespace["_real_numpy"] = np
    with pytest.raises(namespace["CacheError"]):
        namespace["write_cache_members"]("p", {"images": payload, "masks": payload, "singles": payload}, verify=True)
    # nothing was published, so the patient is simply "not cached" and gets rebuilt
    namespace["np"] = np
    assert namespace["cache_layout"]("p") is None
    for part in namespace["parts_of"]("p").values():
        assert not part.exists()


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def test_the_data_pipeline_benchmark_measures_both_readers(archive_cache):
    cache, volumes = archive_cache
    namespace = make_namespace(cache)
    legacy = namespace["benchmark_data_pipeline"](volumes, batch_size=4, size=32, steps=1, warmup=1, legacy=True)
    optimised = namespace["benchmark_data_pipeline"](volumes, batch_size=4, size=32, steps=1, warmup=1)
    for report in (legacy, optimised):
        assert report["first_batch_seconds"] is not None and report["first_batch_seconds"] >= 0
        assert report["batch_seconds"]["n"] >= 1
        assert report["batch_seconds"]["median"] >= 0
        assert report["index_seconds"] >= 0
        assert report["slices_available"] == SLICES * len(volumes)
    assert legacy["pipeline"] == "legacy" and optimised["pipeline"] == "cached-array"
    # the legacy index build loads a whole payload per patient; the new one reads headers and
    # then only the members a batch asks for (images + masks, never `singles`)
    assert legacy["store"]["member_reads"] >= len(volumes) * 3
    assert optimised["store"]["member_reads"] <= 2 * len(volumes)
    assert optimised["index_seconds"] <= legacy["index_seconds"]
