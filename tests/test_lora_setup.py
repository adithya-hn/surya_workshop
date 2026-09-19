"""Tests for the LoRA fine-tuning setup.

These guard two bugs that made every ``use_lora: true`` run meaningless:

1. The fine-tuning head was frozen at its random initialisation, because
   ``apply_peft_lora()`` passed no ``modules_to_save``.
2. ``target_modules`` named layers (``q_proj``/``k_proj``/``v_proj``/
   ``out_proj``) that do not exist in the Surya backbone, which uses a fused
   ``attn.qkv``.  PEFT only errors when *no* entry matches, so the attention
   layers were silently never adapted.

Everything runs on CPU with a tiny backbone, so the suite is fast.
"""

import pytest
import torch
from torch import nn

from conftest import (
    DEPTH,
    EMBED_DIM,
    IMG_SIZE,
    IN_CHANS,
    N_ATTENTION_BLOCKS,
    N_SPECTRAL_BLOCKS,
    PATCH_SIZE,
    make_batch,
    make_model,
)
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.models.finetune_models import ClassToken
from workshop_infrastructure.utils import (
    _NORM_LAYERS,
    HEAD_PREFIX,
    apply_peft_lora,
    cast_frozen_to,
    disable_peft_input_dtype_cast,
    discover_head_modules,
)


def adapted_modules(peft_model):
    """Qualified names of the modules PEFT wrapped with a LoRA adapter."""
    return {
        name.split(".lora_A")[0].replace("base_model.model.", "")
        for name, _ in peft_model.named_parameters()
        if ".lora_A" in name
    }


# ---------------------------------------------------------------------------
# target_modules
# ---------------------------------------------------------------------------


def test_adapted_modules_are_exactly_the_intended_set():
    """fc1/fc2 in every block, plus attn.qkv/attn.proj in the attention blocks."""
    model = apply_peft_lora(make_model(), LoraAdapterConfig())

    expected = set()
    for i in range(N_SPECTRAL_BLOCKS):
        prefix = f"backbone.backbone.blocks_spectral_gating.{i}"
        expected |= {f"{prefix}.mlp.fc1", f"{prefix}.mlp.fc2"}
    for i in range(N_ATTENTION_BLOCKS):
        prefix = f"backbone.backbone.blocks_attention.{i}"
        expected |= {
            f"{prefix}.mlp.fc1",
            f"{prefix}.mlp.fc2",
            f"{prefix}.attn.qkv",
            f"{prefix}.attn.proj",
        }

    assert adapted_modules(model) == expected


def test_patch_embedding_and_head_are_never_adapted():
    adapted = adapted_modules(apply_peft_lora(make_model(), LoraAdapterConfig()))
    assert not [n for n in adapted if "embedding" in n], "tokeniser must not be adapted"
    assert not [n for n in adapted if n.startswith(HEAD_PREFIX)], "head must not be adapted"


def test_to_dynamic_projection_is_never_adapted():
    adapted = adapted_modules(apply_peft_lora(make_model(), LoraAdapterConfig()))
    assert not [n for n in adapted if "to_dynamic_projection" in n]


def test_default_target_modules_match_the_backbone():
    """Regression guard: the old split-QKV names match nothing in this backbone."""
    defaults = LoraAdapterConfig().target_modules
    assert defaults == ["fc1", "fc2", "attn.qkv", "attn.proj"]

    module_names = [name for name, _ in make_model().named_modules()]
    for entry in defaults:
        assert any(
            name == entry or name.endswith("." + entry) for name in module_names
        ), f"target_modules entry {entry!r} matches no module; PEFT would ignore it silently"


def test_bare_proj_would_capture_the_tokeniser():
    """Why the dotted 'attn.proj' form is required rather than a bare 'proj'."""
    cfg = LoraAdapterConfig(target_modules=["fc1", "fc2", "qkv", "proj"])
    adapted = adapted_modules(apply_peft_lora(make_model(), cfg))
    assert "backbone.embedding.patch_embed.proj" in adapted


# ---------------------------------------------------------------------------
# The head stays trainable
# ---------------------------------------------------------------------------


def test_head_is_trainable_and_backbone_is_not():
    model = apply_peft_lora(make_model(), LoraAdapterConfig())

    for name, param in model.named_parameters():
        is_adapter = ".lora_" in name
        # PEFT keeps a frozen original alongside the trainable copy.
        is_trainable_head_copy = "modules_to_save" in name

        if is_adapter or is_trainable_head_copy:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_every_head_module_has_a_trainable_copy():
    model = make_model()
    expected = set(discover_head_modules(model))
    assert expected == {"head_cls_token", "head_linear", "head_unembed"}

    peft_model = apply_peft_lora(model, LoraAdapterConfig())
    saved = {
        name.replace("base_model.model.", "").split(".modules_to_save")[0]
        for name, _ in peft_model.named_parameters()
        if "modules_to_save" in name
    }
    assert saved == expected


def test_parameter_free_head_modules_are_not_duplicated():
    """head_dropout carries no parameters, so PEFT need not wrap it."""
    model = make_model()
    assert isinstance(model.head_dropout, nn.Dropout) or model.head_dropout is None
    assert "head_dropout" not in discover_head_modules(model)


# ---------------------------------------------------------------------------
# An optimizer step actually moves the right tensors
# ---------------------------------------------------------------------------


def test_optimizer_step_updates_head_and_lora_b_only():
    torch.manual_seed(0)
    model = apply_peft_lora(make_model(), LoraAdapterConfig())

    tracked = {
        name: param
        for name, param in model.named_parameters()
        if "modules_to_save" in name or ".lora_B" in name
    }
    frozen = {
        name: param
        for name, param in model.named_parameters()
        if name.endswith("attn.qkv.base_layer.weight") or name.endswith("mlp.fc1.base_layer.weight")
    }
    assert tracked and frozen

    before = {name: param.detach().clone() for name, param in {**tracked, **frozen}.items()}

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    model(make_batch()).sum().backward()
    optimizer.step()

    # lora_B starts at zero, so it only moves if gradient reaches it through the head.
    for name, param in tracked.items():
        assert not torch.equal(param, before[name]), f"{name} did not change"
    for name, param in frozen.items():
        assert torch.equal(param, before[name]), f"frozen {name} changed"


def test_class_token_receives_gradient():
    """The specific symptom of the original bug: cls_token stuck at zeros."""
    model = apply_peft_lora(make_model(pooling="class_token"), LoraAdapterConfig())
    token = dict(model.named_parameters())[
        "base_model.model.head_cls_token.modules_to_save.default.token"
    ]
    assert torch.count_nonzero(token) == 0, "class_token should start at zeros"

    model(make_batch()).sum().backward()
    assert token.grad is not None and torch.count_nonzero(token.grad) > 0


@pytest.mark.parametrize(
    "pooling", ["class_token", "transformer", "attention", "global_average"]
)
def test_all_poolings_build_and_train_one_step(pooling):
    torch.manual_seed(0)
    model = apply_peft_lora(make_model(pooling=pooling), LoraAdapterConfig())

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    output = model(make_batch())
    assert output.shape == (2,)
    output.sum().backward()
    optimizer.step()

    head_params = [p for n, p in model.named_parameters() if "modules_to_save" in n]
    assert head_params, f"{pooling} produced no trainable head parameters"
    assert all(p.grad is not None for p in head_params)


# ---------------------------------------------------------------------------
# Validation of the head_ convention
# ---------------------------------------------------------------------------


def test_head_module_without_prefix_is_rejected():
    model = make_model()
    model.extra_head = nn.Linear(EMBED_DIM, 1)  # missing the head_ prefix

    with pytest.raises(ValueError, match="extra_head"):
        discover_head_modules(model)


def test_parameter_free_module_without_prefix_is_allowed():
    model = make_model()
    model.some_dropout = nn.Dropout(0.1)  # no parameters -> exempt
    assert "some_dropout" not in discover_head_modules(model)


def test_bare_top_level_parameter_is_rejected():
    model = make_model()
    model.head_raw_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))

    with pytest.raises(ValueError, match="head_raw_token"):
        discover_head_modules(model)


def test_head_name_colliding_with_backbone_is_rejected():
    """PEFT matches modules_to_save with a bare endswith, so suffixes collide."""
    model = make_model()
    model.backbone.custom_linear = nn.Linear(EMBED_DIM, EMBED_DIM)

    # "head_linear" is not a suffix of "backbone.custom_linear", but "linear" is
    # -- reproduce the hazard with a name that really does collide.
    model.backbone.my_head_linear = nn.Linear(EMBED_DIM, EMBED_DIM)

    with pytest.raises(ValueError, match="collides"):
        discover_head_modules(model)


# ---------------------------------------------------------------------------
# ClassToken
# ---------------------------------------------------------------------------


def test_class_token_expands_to_batch_size():
    token = ClassToken(EMBED_DIM)
    assert token(1).shape == (1, 1, EMBED_DIM)
    assert token(5).shape == (5, 1, EMBED_DIM)


def test_class_token_forward_dispatches_to_trainable_copy_under_peft():
    """Calling the module must reach the trainable copy, not the frozen original."""
    model = apply_peft_lora(make_model(pooling="class_token"), LoraAdapterConfig())
    wrapper = model.base_model.model.head_cls_token

    output = wrapper(3)
    assert output.shape == (3, 1, EMBED_DIM)
    assert output.requires_grad, "token read must be differentiable"

    output.sum().backward()
    assert wrapper.modules_to_save["default"].token.grad is not None
    assert wrapper.original_module.token.grad is None


def test_class_token_init_modes():
    torch.manual_seed(0)
    assert torch.count_nonzero(ClassToken(EMBED_DIM, init="zeros").token) == 0
    assert torch.count_nonzero(ClassToken(EMBED_DIM, init="randn").token) > 0
    with pytest.raises(ValueError):
        ClassToken(EMBED_DIM, init="uniform")


# ---------------------------------------------------------------------------
# Memory helpers: disable_peft_input_dtype_cast / cast_frozen_to
# ---------------------------------------------------------------------------


def test_disable_peft_input_dtype_cast_touches_every_tuner_layer():
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    from peft.tuners.tuners_utils import BaseTunerLayer

    expected = sum(1 for m in model.modules() if isinstance(m, BaseTunerLayer))
    assert expected > 0, "no tuner layers to disable -- the fixture is wrong, not the code"

    touched = disable_peft_input_dtype_cast(model)

    assert touched == expected
    assert all(
        m.cast_input_dtype_enabled is False
        for m in model.modules()
        if isinstance(m, BaseTunerLayer)
    )


def test_disable_peft_input_dtype_cast_stops_the_fp32_upcast():
    """The point of the helper: under autocast the adapter input stays in half.

    This is the allocation that OOM'd the sf_triggers 2D app -- PEFT upcast the activation
    to the fp32 adapter dtype, and at 4096x4096 mlp.fc2's input is (1, 65536, 5120), so the
    wasted copy was 1.25 GiB per block per gradient-checkpoint recompute.
    """
    seen = {}

    def record(name):
        def hook(_module, args, _output):
            seen[name] = args[0].dtype
        return hook

    for label, disable in (("on", False), ("off", True)):
        model = apply_peft_lora(make_model(), LoraAdapterConfig())
        if disable:
            disable_peft_input_dtype_cast(model)
        target = dict(model.named_modules())[
            "base_model.model.backbone.backbone.blocks_attention.0.mlp.fc2.lora_A.default"
        ]
        target.register_forward_hook(record(label))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(make_batch(batch_size=1))

    assert seen["on"] == torch.float32     # current behaviour: the wasteful upcast
    assert seen["off"] == torch.bfloat16   # after the fix: no copy at all


def test_cast_frozen_to_leaves_trainable_params_in_fp32():
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    n_params, n_buffers = cast_frozen_to(model, torch.float16)

    assert n_params > 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "nothing trainable -- the fixture is wrong"
    assert all(p.dtype is torch.float32 for p in trainable), (
        "trainable parameters must stay fp32 so the optimizer keeps a master copy"
    )

    # Everything frozen is half, except normalization layers, which are held back on
    # purpose -- see test_cast_frozen_to_keeps_norm_layers_in_fp32.
    norm_params = {
        id(p)
        for m in model.modules()
        if isinstance(m, _NORM_LAYERS)
        for p in m.parameters(recurse=False)
    }
    frozen_non_norm = [
        p for p in model.parameters() if not p.requires_grad and id(p) not in norm_params
    ]
    assert frozen_non_norm
    assert all(p.dtype is torch.float16 for p in frozen_non_norm)


def test_cast_frozen_to_keeps_norm_layers_in_fp32():
    """LayerNorm must stay fp32: F.layer_norm on CPU refuses a half weight.

    A half-precision LayerNorm weight raises "mixed dtype (CPU): expect parameter to have
    scalar type of Float" against an fp32 input -- which is what the notebooks' pre-GPU
    smoke test does. CUDA tolerates it, so this only shows up off the accelerator. Norm
    params are 2 * embed_dim each, so holding them back costs almost nothing.
    """
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    cast_frozen_to(model, torch.float16)

    norms = [m for m in model.modules() if isinstance(m, _NORM_LAYERS)]
    assert norms, "fixture has no normalization layers to check"
    for norm in norms:
        for param in norm.parameters(recurse=False):
            assert param.dtype is torch.float32

    # And the whole point: a CPU forward under autocast still runs.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(make_batch(batch_size=1))


def test_cast_frozen_to_casts_the_pos_embed_buffer():
    """pos_embed is a buffer, not a parameter, and it is why this helper exists.

    embedding.py adds it to the half-precision patch-embedding output; fp32 + half promotes
    back to fp32, which pins the whole inter-block residual stream to fp32. Leaving buffers
    alone would forfeit most of the saving.
    """
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    pos_embed = dict(model.named_buffers())[
        "base_model.model.backbone.embedding.pos_embed"
    ]
    assert pos_embed.dtype is torch.float32

    cast_frozen_to(model, torch.float16)

    assert dict(model.named_buffers())[
        "base_model.model.backbone.embedding.pos_embed"
    ].dtype is torch.float16


def test_cast_frozen_to_is_idempotent():
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    cast_frozen_to(model, torch.float16)
    n_params, n_buffers = cast_frozen_to(model, torch.float16)
    assert (n_params, n_buffers) == (0, 0)


def test_model_still_runs_a_step_after_both_helpers():
    """End to end: half frozen weights + fp32 adapters must train under autocast."""
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    disable_peft_input_dtype_cast(model)
    cast_frozen_to(model, torch.bfloat16)

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(make_batch()).float().square().mean()
    loss.backward()
    optimizer.step()

    after = [p for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "no trainable parameter moved -- the fp32 master copy is not being updated"
    )


def test_2d_head_trains_after_both_helpers():
    """The 2D decoder path is the one that OOM'd, so pin it separately from the 1D fixture.

    HelioSpectformer2D keeps the full token grid and feeds it to a LinearDecoder
    (Conv2d + PixelShuffle), rather than collapsing to a single CLS token as the 1D wrapper
    does. Both helpers have to leave that path trainable and runnable under autocast.
    """
    from workshop_infrastructure.models.finetune_models import HelioSpectformer2D

    model = HelioSpectformer2D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        time_embedding={"type": "linear", "time_dim": 1},
        depth=DEPTH,
        n_spectral_blocks=N_SPECTRAL_BLOCKS,
        num_heads=2,
        mlp_ratio=4,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dtype=torch.float32,
        ft_out_chans=1,
    )
    model = apply_peft_lora(model, LoraAdapterConfig())
    disable_peft_input_dtype_cast(model)
    cast_frozen_to(model, torch.float16)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(make_batch())
    assert out.shape == (2, 1, IMG_SIZE, IMG_SIZE), "decoder must return full-resolution maps"

    head_weight = dict(model.named_parameters())[
        "base_model.model.head_unembed.modules_to_save.default.unembed.0.weight"
    ]
    assert head_weight.requires_grad and head_weight.dtype is torch.float32
    before = head_weight.detach().clone()

    out.float().square().mean().backward()
    optimizer.step()

    assert not torch.equal(before, head_weight), "the decoder head did not train"


def test_checkpoint_head_still_produces_head_gradients():
    """ft_checkpoint_head trades recompute for memory; it must not drop gradients.

    use_reentrant=False matters here: the reentrant checkpoint variant silently produces no
    gradients when none of the checkpointed region's inputs require grad, which is exactly
    the case under a frozen backbone.
    """
    from workshop_infrastructure.models.finetune_models import HelioSpectformer2D

    def build(checkpoint_head):
        model = HelioSpectformer2D(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANS,
            embed_dim=EMBED_DIM,
            time_embedding={"type": "linear", "time_dim": 1},
            depth=DEPTH,
            n_spectral_blocks=N_SPECTRAL_BLOCKS,
            num_heads=2,
            mlp_ratio=4,
            drop_rate=0.0,
            window_size=2,
            dp_rank=2,
            dtype=torch.float32,
            ft_out_chans=1,
            ft_checkpoint_head=checkpoint_head,
        )
        return apply_peft_lora(model, LoraAdapterConfig())

    torch.manual_seed(0)
    batch = make_batch()

    grads = {}
    for label, flag in (("plain", False), ("checkpointed", True)):
        model = build(flag)
        model.train()
        model(batch).float().square().mean().backward()
        grads[label] = {
            n: p.grad for n, p in model.named_parameters() if p.requires_grad
        }
        assert all(g is not None for g in grads[label].values()), (
            f"{label}: some trainable parameter received no gradient"
        )

    assert grads["plain"].keys() == grads["checkpointed"].keys()
