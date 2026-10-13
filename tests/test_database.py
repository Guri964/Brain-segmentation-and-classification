from __future__ import annotations
def test_schema_has_no_identifying_columns(services):
    repository = services["repository"]
    with repository.connection() as connection:
        columns = {
            row["name"]
            for table in ("analyses", "scan_metadata")
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
    for forbidden in ("patient_name", "name", "mrn", "date_of_birth", "dob", "address"):
        assert forbidden not in columns
def test_analysis_round_trip_and_delete(services):
    repository = services["repository"]
    analysis_id = "an-test-0001"
    repository.create_analysis(
        analysis_id=analysis_id,
        scan_identifier="TEST-001",
        analysis_type="single_channel",
        model_ids=["unet_dynamic_512"],
        device="cpu",
        input_manifest={"files": ["t.png"]},
        preprocessing={"pipeline": "single_channel_512"},
    )
    assert repository.get_analysis(analysis_id)["scan_identifier"] == "TEST-001"
    repository.add_segmentation_result(
        analysis_id,
        {
            "model_id": "unet_dynamic_512",
            "display_name": "U-Net",
            "threshold": 0.5,
            "foreground_pixels": 10,
            "image_pixels": 100,
            "foreground_fraction": 0.1,
            "probability_mean": 0.6,
            "probability_max": 0.9,
            "inference_seconds": 0.5,
            "tta_used": False,
            "mc_dropout_used": False,
            "metrics": {},
            "artifact_paths": {"mask": "mask_unet_dynamic_512.png"},
            "warnings": [],
        },
    )
    full = repository.get_full_analysis(analysis_id)
    assert len(full["segmentation"]) == 1
    repository.delete_analysis(analysis_id)
    assert repository.get_analysis(analysis_id) is None
def test_duplicate_analysis_id_is_rejected(services):
    import sqlite3
    repository = services["repository"]
    payload = {
        "scan_identifier": "TEST-002",
        "analysis_type": "single_channel",
        "model_ids": [],
        "device": "cpu",
        "input_manifest": {"files": []},
        "preprocessing": {},
    }
    repository.create_analysis(analysis_id="an-test-0002", **payload)
    try:
        repository.create_analysis(analysis_id="an-test-0002", **payload)
    except sqlite3.IntegrityError:
        return
    raise AssertionError("a duplicate analysis identifier was accepted")
def test_search_and_filter(services):
    repository = services["repository"]
    repository.create_analysis(
        analysis_id="an-test-0003",
        scan_identifier="SEARCHABLE-42",
        analysis_type="four_channel",
        model_ids=["segnet_brats_4ch"],
        device="cpu",
        input_manifest={"files": []},
        preprocessing={},
        status="completed",
    )
    found = repository.list_analyses(search="SEARCHABLE", limit=10, offset=0)
    assert any(item["analysis_id"] == "an-test-0003" for item in found["items"])
    by_type = repository.list_analyses(analysis_type="four_channel", limit=10, offset=0)
    assert all(item["analysis_type"] == "four_channel" for item in by_type["items"])
def test_statistics_shape(services):
    payload = services["repository"].statistics()
    assert {"totals", "recent"} <= set(payload)
    assert {"analyses", "segmentations", "radiomics", "reports", "classifications"} <= set(payload["totals"])
