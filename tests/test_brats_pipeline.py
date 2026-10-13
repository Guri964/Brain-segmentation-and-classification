from __future__ import annotations

import json

import numpy as np
import pytest


def _write_slice(path, *, slice_index: int, empty: bool = False) -> None:
    import h5py

    height = width = 32
    image = np.zeros((height, width, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for channel in range(4):
        image[:, :, channel] = np.exp(
            -(((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * (6.0 + channel) ** 2))
        )
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    if not empty:
        mask[10:22, 10:22, 0] = 1
        mask[12:20, 12:20, 1] = 1
        mask[14:18, 14:18, 2] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=image)
        handle.create_dataset("mask", data=mask)
        handle.attrs["slice_index"] = slice_index


@pytest.fixture()
def slice_tree(tmp_path):
    root = tmp_path / "BraTS2020_training_data" / "content" / "data"
    for volume in ("volume_1", "volume_2"):
        for index in range(3):
            _write_slice(
                root / volume / f"{volume}_slice_{index}.h5",
                slice_index=index,
                empty=(volume == "volume_2" and index == 2),
            )
    mapping = tmp_path / "name_mapping.csv"
    mapping.write_text(
        "Grade,BrATS_2020_subject_ID\n"
        "HGG,volume_1\n"
        "LGG,volume_2\n",
        encoding="utf-8",
    )
    return root, mapping


def test_detect_keys_and_read_slice(slice_tree):
    import h5py

    from brats import detect_keys, read_slice_file

    root, _ = slice_tree
    path = root / "volume_1" / "volume_1_slice_0.h5"
    with h5py.File(path, "r") as handle:
        image_key, mask_key = detect_keys(list(handle.keys()))
    assert image_key == "image"
    assert mask_key == "mask"

    slice_obj = read_slice_file(path, image_key=image_key, mask_key=mask_key)
    assert slice_obj.channels == 4
    assert slice_obj.mask_semantics["kind"] in {
        "one_hot_channels",
        "brats_label_map",
        "multilabel",
    }
    assert int(slice_obj.whole_tumour().sum()) > 0
    assert slice_obj.image_key == "image"


def test_classify_mask_never_invents_sub_region_identity():
    from brats import classify_mask

    one_hot = classify_mask(np.stack([np.ones((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)]))
    assert one_hot["semantics"]["kind"] == "one_hot_channels"
    assert "channel_labels" not in one_hot["semantics"]
    assert one_hot["semantics"]["channel_labels_declared"] is False

    declared = classify_mask(
        np.stack([np.ones((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)]),
        declared_channel_labels=[1, 2],
    )
    assert declared["semantics"]["channel_labels"] == [1, 2]

    label_map = classify_mask(np.array([[0, 1], [2, 4]], dtype=np.uint8))
    assert label_map["semantics"]["kind"] in {"brats_label_map", "multilabel"}

    absent = classify_mask(np.zeros((4, 4), dtype=np.uint8))
    assert absent["semantics"]["kind"] == "binary"
    assert int(absent["array"].sum()) == 0


def test_discover_and_group_slice_files(slice_tree):
    from brats import discover_slice_files, group_by_volume

    root, _ = slice_tree
    files = discover_slice_files(root)
    assert len(files) == 6
    grouped = group_by_volume(files)
    assert len(grouped) == 2
    assert all(len(entries) == 3 for entries in grouped.values())


def test_grade_table_column_selection_and_unknown_values(slice_tree):
    from brats import normalise_grade, read_grade_table

    _, mapping = slice_tree
    table = read_grade_table(mapping)
    assert {entry["grade"] for entry in table.values()} == {"HGG", "LGG"}
    assert read_grade_table(mapping, column="Grade")
    with pytest.raises(ValueError):
        read_grade_table(mapping, column="does_not_exist")
    assert normalise_grade("not a grade") is None


def _manifest(tmp_path, slice_tree, *extra: str):
    from _common import load_manifest
    from prepare_brats import main

    root, mapping = slice_tree
    out = tmp_path / "manifest.json"
    argv = ["--root", str(root), "--out", str(out), "--grade-table", str(mapping), "--splits", *extra]
    assert main(argv) == 0
    return out, load_manifest(out)


def test_manifest_is_patient_level_labelled_and_leakage_free(tmp_path, slice_tree):
    out, dataset = _manifest(tmp_path, slice_tree)
    assert dataset.splits
    assert dataset.summary()["patients"] == 2
    for case in dataset.cases:
        assert case.source_kind == "brats_hdf5_slice"
        assert case.options.get("mask_source") == "same_file"
        assert case.patient_id
        assert case.patient_id != case.case_id
    known = {case.case_id for case in dataset.cases}
    for name, ids in dataset.splits.items():
        assert set(ids) <= known, f"split '{name}' lists ids that are not cases: {ids}"
    metadata = json.loads(out.read_text(encoding="utf-8"))["split_metadata"]
    assert metadata["leakage_free"] is True
    assert metadata["patient_overlap"] == []
    assert metadata["method"] == "patient_level_stratified_by_label"
    assert metadata["cases_assigned"] == len(dataset.cases)


def test_manifest_labels_come_from_the_grade_table(tmp_path, slice_tree):
    _, dataset = _manifest(tmp_path, slice_tree)
    labels = {case.patient_id: case.label for case in dataset.cases}
    assert set(labels.values()) == {"HGG", "LGG"}
    assert all(case.options.get("mask_semantics") for case in dataset.cases)


def test_empty_mask_slices_are_dropped_by_default(tmp_path, slice_tree):
    _, dataset = _manifest(tmp_path, slice_tree)
    assert all(case.case_id != "volume_2_2" for case in dataset.cases)
    _, kept = _manifest(tmp_path / "keep", slice_tree, "--slice-filter", "all")
    assert any(case.case_id == "volume_2_2" for case in kept.cases)


def test_cases_load_from_disk_as_channel_first_arrays(tmp_path, slice_tree):
    from _training import _ArrayLoader, load_case_arrays

    _, dataset = _manifest(tmp_path, slice_tree)
    case = next(item for item in dataset.cases if item.case_id.startswith("volume_1"))
    image, mask = load_case_arrays(case, 16, 4)
    assert image.shape == (4, 16, 16)
    assert mask.shape == (16, 16)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.max() == 1.0
    assert np.isfinite(image).all()

    pairs = [load_case_arrays(item, 16, 4) for item in dataset.cases[:3]]
    images = np.stack([pair[0] for pair in pairs]).astype(np.float32)
    masks = np.stack([pair[1] for pair in pairs]).astype(np.float32)
    loader = _ArrayLoader(images, masks, 2, shuffle=False, seed=0)
    assert len(loader) == 2
    batch_images, batch_masks = next(iter(loader))
    assert batch_images.shape == (2, 4, 16, 16)
    assert batch_masks.shape == (2, 16, 16)

    augmented = _ArrayLoader(images, masks, 3, shuffle=True, seed=0, augment=True)
    aug_images, aug_masks = next(iter(augmented))
    assert aug_images.shape[1:] == (4, 16, 16)
    assert aug_masks.shape == (3, 16, 16)


def test_inspection_report_is_written_and_files_are_untouched(tmp_path, slice_tree):
    from inspect_brats import main

    root, mapping = slice_tree
    report = tmp_path / "inspection.json"
    before = sorted((path.name, path.stat().st_size) for path in root.rglob("*.h5"))
    exit_code = main(
        [
            "--root", str(root),
            "--grade-table", str(mapping),
            "--grade-column", "Grade",
            "--json", str(report),
        ]
    )
    assert exit_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["tree"]["files"] == 6
    assert payload["tree"]["volumes"] == 2
    assert payload["grade_table"]["class_counts"] == {"HGG": 1, "LGG": 1}
    after = sorted((path.name, path.stat().st_size) for path in root.rglob("*.h5"))
    assert before == after


def test_inspection_rejects_an_unknown_grade_column(tmp_path, slice_tree):
    from inspect_brats import main

    root, mapping = slice_tree
    with pytest.raises(ValueError):
        main(["--root", str(root), "--grade-table", str(mapping), "--grade-column", "nope"])

