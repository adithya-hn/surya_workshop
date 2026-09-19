"""Tests for the config-level cross-field validation and the nglo/pooling fix.

Guards the bug where ``model.nglo`` had to be hand-kept in sync with ``model.pooling`` or
the long-short attention reshape failed deep in vendored code, on the first forward pass
(not at construction). ``nglo`` is now derived from ``pooling`` inside
``HelioSpectformer1D`` rather than being a config field, so this also guards the other
config-level invariants added alongside it: ``ModelConfig.__post_init__`` and the
cross-section checks in ``TrainingConfig.__post_init__``.

Everything runs on CPU with a tiny backbone, so the suite is fast.
"""

import dataclasses

import pytest

from conftest import make_batch, make_model
from downstream_apps.template.configs import load_flare_config
from workshop_infrastructure.configs import (
    DataConfig,
    ModelConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
    VALID_PRECISION,
    _TRAINING_KEYS,
)


def make_data_config(**overrides):
    kwargs = dict(
        train_data_path="train.csv",
        valid_data_path="valid.csv",
        scalers_path="scalers.yaml",
        channels=["aia171"],
        time_delta_input_minutes=[0],
        time_delta_target_minutes=60,
    )
    kwargs.update(overrides)
    return DataConfig(**kwargs)


def make_training_config(**overrides):
    kwargs = dict(job_id="test", data=make_data_config(), model=ModelConfig())
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)


# ---------------------------------------------------------------------------
# nglo is derived, not configured
# ---------------------------------------------------------------------------


def test_model_config_has_no_nglo_field():
    assert "nglo" not in {f.name for f in dataclasses.fields(ModelConfig)}


def test_shipped_config_loads_and_has_no_nglo():
    cfg = load_flare_config("downstream_apps/template/configs/config_script.yaml")
    assert not hasattr(cfg.model, "nglo")


@pytest.mark.parametrize("pooling", ["class_token", "transformer", "attention", "global_average"])
def test_every_pooling_runs_a_forward_pass_with_the_derived_nglo(pooling):
    """The original bug only surfaced in forward(), not at construction."""
    model = make_model(pooling=pooling)
    output = model(make_batch())
    assert output.shape == (2,)


# ---------------------------------------------------------------------------
# ModelConfig.__post_init__
# ---------------------------------------------------------------------------


def test_default_model_config_is_valid():
    ModelConfig()  # must not raise


def test_img_size_not_divisible_by_patch_size_raises():
    with pytest.raises(ValueError, match="img_size"):
        ModelConfig(img_size=100, patch_size=16)


def test_img_size_divisible_by_patch_size_is_accepted():
    ModelConfig(img_size=64, patch_size=16)


def test_spectral_blocks_greater_than_depth_raises():
    with pytest.raises(ValueError, match="spectral_blocks"):
        ModelConfig(depth=3, spectral_blocks=4)


@pytest.mark.parametrize("spectral_blocks", [0, 3])
def test_spectral_blocks_at_the_boundary_is_accepted(spectral_blocks):
    ModelConfig(depth=3, spectral_blocks=spectral_blocks, checkpoint_layers=[])


def test_checkpoint_layers_out_of_range_raises():
    with pytest.raises(ValueError, match="checkpoint_layers"):
        ModelConfig(depth=3, checkpoint_layers=[0, 3])


def test_checkpoint_layers_negative_raises():
    with pytest.raises(ValueError, match="checkpoint_layers"):
        ModelConfig(depth=3, checkpoint_layers=[-1])


def test_checkpoint_layers_in_range_is_accepted():
    ModelConfig(depth=3, checkpoint_layers=[0, 1, 2])


def test_learned_flow_with_non_linear_time_embedding_raises():
    with pytest.raises(ValueError, match="learned_flow"):
        ModelConfig(learned_flow=True, time_embedding=TimeEmbeddingConfig(type="perceiver"))


def test_learned_flow_with_linear_time_embedding_is_accepted():
    ModelConfig(learned_flow=True, time_embedding=TimeEmbeddingConfig(type="linear"))


# ---------------------------------------------------------------------------
# TrainingConfig.__post_init__ (cross-section checks)
# ---------------------------------------------------------------------------


def test_deterministic_true_with_learned_flow_raises():
    with pytest.raises(ValueError, match="learned_flow"):
        make_training_config(
            model=ModelConfig(learned_flow=True), deterministic=True,
        )


def test_deterministic_warn_with_learned_flow_is_accepted():
    make_training_config(model=ModelConfig(learned_flow=True), deterministic="warn")


def test_time_dim_greater_than_available_deltas_raises():
    with pytest.raises(ValueError, match="time_dim"):
        make_training_config(
            data=make_data_config(time_delta_input_minutes=[0, -60]),
            model=ModelConfig(time_embedding=TimeEmbeddingConfig(time_dim=3)),
        )


def test_time_dim_less_than_or_equal_to_available_deltas_is_accepted():
    make_training_config(
        data=make_data_config(time_delta_input_minutes=[0, -60]),
        model=ModelConfig(time_embedding=TimeEmbeddingConfig(time_dim=2)),
    )


# ---------------------------------------------------------------------------
# training.precision / training.accumulate_grad_batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", VALID_PRECISION)
def test_every_valid_precision_is_accepted(precision):
    assert make_training_config(precision=precision).precision == precision


def test_unknown_precision_raises_and_names_the_alternatives():
    with pytest.raises(ValueError, match="training.precision") as excinfo:
        make_training_config(precision="fp16")
    # The error has to be actionable: a typo should tell you what to write instead.
    for valid in VALID_PRECISION:
        assert valid in str(excinfo.value)


def test_default_precision_is_mixed_so_cast_frozen_to_stays_reachable():
    # build_model() only calls cast_frozen_to() under a *-mixed precision, because half
    # weights need autocast to reconcile them with the fp32 adapters. If the default ever
    # changes to a *-true or 32-true mode, that memory lever silently stops firing.
    assert make_training_config().precision.endswith("-mixed")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_accumulate_grad_batches_rejects_non_positive_ints(bad):
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        make_training_config(accumulate_grad_batches=bad)


def test_accumulate_grad_batches_accepts_positive_ints():
    assert make_training_config(accumulate_grad_batches=4).accumulate_grad_batches == 4


def test_training_keys_whitelist_matches_the_dataclass_fields():
    """Guard the two-place edit that adding a training: key requires.

    A key in _TRAINING_KEYS without a matching TrainingConfig field silently does nothing;
    a field without the key makes the YAML raise on a legitimate value. Neither shows up
    until someone edits a config, so pin the correspondence here instead.
    """
    field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
    # These live on TrainingConfig but come from other sections of the YAML.
    not_from_training_section = {
        "job_id", "data", "model", "output", "wandb_project", "wandb_entity",
    }
    assert _TRAINING_KEYS <= field_names
    assert field_names - not_from_training_section == set(_TRAINING_KEYS)


def test_unknown_training_key_still_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "job_id: test\n"
        "data:\n"
        "  train_data_path: train.csv\n"
        "  valid_data_path: valid.csv\n"
        "  scalers_path: scalers.yaml\n"
        "  channels: [aia171]\n"
        "  time_delta_input_minutes: [0]\n"
        "  time_delta_target_minutes: 60\n"
        "model: {}\n"
        "training:\n"
        "  precission: 16-mixed\n"  # deliberate typo
    )
    with pytest.raises(ValueError, match="precission"):
        load_flare_config(str(cfg))
