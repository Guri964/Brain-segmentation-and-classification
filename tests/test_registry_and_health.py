from __future__ import annotations
def test_registry_lists_every_descriptor(services):
    registry = services["registry"]
    assert registry.descriptors, "no descriptor is registered"
    for descriptor in registry.descriptors.values():
        payload = descriptor.to_dict()
        assert "input_spec" in payload and "output_semantics" in payload
def test_missing_checkpoint_reports_a_message(services):
    loaded = services["registry"].load_model("raunet_brats_4ch")
    assert loaded.is_ready is False
    assert "raise" in loaded.message.lower() or "generate" in loaded.message.lower() or "train" in loaded.message.lower()
def test_health_check_is_deterministic(services):
    first = services["health"].check("unet_dynamic_512")
    second = services["health"].check("unet_dynamic_512")
    assert first.verdict == second.verdict
    assert first.phantom_version == second.phantom_version
def test_unavailable_model_is_not_reported_as_responsive(services):
    health = services["health"].check("radiomics_mlp_v1")
    assert health.responsive is False
    assert health.status in {"not_loaded", "incompatible", "failed"}
def test_health_summary_shape(services):
    summary = services["health"].summary()
    assert {"phantom_version", "generated_at", "models", "responsive_models"} <= set(summary)
    for entry in summary["models"]:
        assert entry["verdict"]
        assert isinstance(entry["responsive"], bool)
