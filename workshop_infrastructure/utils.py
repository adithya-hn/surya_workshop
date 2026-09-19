import functools
import os
import logging
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from workshop_infrastructure.configs import LoraAdapterConfig

# Optional: S3 helpers below degrade to a clear ImportError when boto3 is absent.
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
except Exception:  # pragma: no cover
    boto3 = None
    UNSIGNED = None
    BotoConfig = None


# ---------------------------------------------------------------------------
# AWS / infrastructure utilities
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def detect_ec2_region() -> str | None:
    """Return the AWS region of the current EC2 instance, or None if not on EC2.

    Queries the IMDSv2 endpoint (169.254.169.254), which is only reachable from
    within an EC2 instance.  The 1-second timeout makes this a no-op on any
    other machine.  Results are cached so the network round-trip happens at most
    once per process.
    """
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(token_req, timeout=1) as resp:
            token = resp.read().decode()
        region_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(region_req, timeout=1) as resp:
            return resp.read().decode()
    except Exception:
        return None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Raises:
        ValueError: If ``uri`` is not an S3 URI, or has no key component.
    """
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    remainder = uri[len("s3://"):]
    if "/" not in remainder:
        raise ValueError(f"S3 URI is missing a key: {uri!r} (expected s3://bucket/key)")
    bucket, key = remainder.split("/", 1)
    return bucket, key


def make_s3_client(
    anon: bool = False,
    region: str | None = None,
    pool_size: int = 32,
    max_attempts: int = 10,
):
    """Return a configured boto3 S3 client.

    Shared by the dataset loader and the S3 benchmark so both use identical connection
    pooling and retry behaviour.

    Args:
        anon: If True, sign requests anonymously (public buckets).
        region: AWS region. Pass ``None`` to let boto3 resolve it.
        pool_size: Max connections in the client's connection pool. Should be at least
            twice the download concurrency, or threads will contend for connections.
        max_attempts: Retry attempts in adaptive mode. The dataset uses a high value
            because a failed read kills a training run; the benchmark uses a low one so
            throughput measurements are not skewed by retries.
    """
    if boto3 is None:
        raise ImportError("boto3 is required for S3 access. Install via: pip install boto3")

    retry_cfg = {"max_attempts": max_attempts, "mode": "adaptive"}
    if anon:
        config = BotoConfig(
            signature_version=UNSIGNED,
            max_pool_connections=pool_size,
            retries=retry_cfg,
        )
    else:
        config = BotoConfig(max_pool_connections=pool_size, retries=retry_cfg)
    return boto3.client("s3", region_name=region, config=config)


# ---------------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def create_logger(output_dir: str, dist_rank: int, name: str) -> logging.Logger:
    """Create a file+console logger identified by name and rank."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s]: %(levelname)s %(message)s"

    if name.endswith("main"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{name}.log"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Scaler utilities
# ---------------------------------------------------------------------------

def class_from_name(module_name: str, class_name: str):
    """Import and return a class by its module and class name strings."""
    m = __import__(module_name, globals(), locals(), [class_name])
    return getattr(m, class_name)


def build_scalers(info) -> Dict:
    """Reconstruct per-channel scaler objects from a scalers YAML file or dict.

    Args:
        info: Path to a scalers YAML file, or an already-loaded dict mapping
            channel name -> scaler parameters.

    Returns:
        Dict mapping channel name -> scaler instance (e.g. ``StandardScaler``), each
        exposing ``.mean``, ``.std``, ``.epsilon`` and ``.sl_scale_factor``.

    On the ``base:`` field
    ---------------------
    Every entry records the module the scaler was originally fitted in — in the
    shipped ``scalers.yaml`` that is ``surya.datasets.transformations``, which does not
    exist in this repo (the Surya code is vendored under ``workshop_infrastructure/``).

    That field is deliberately **ignored**. Classes are always resolved from
    ``workshop_infrastructure.datasets.transformations``, never imported from the
    recorded path, so normalization does not depend on what happens to be installed in
    the environment: were a real ``surya`` package ever added, honoring ``base`` would
    silently switch which ``StandardScaler`` implementation runs, and results could move
    because of an unrelated install. Determinism matters more here than deference to a
    stale field. This is not warned about at runtime — the field is stale in every entry
    of a file fetched from HuggingFace that nobody can edit, so a warning would fire
    always and invite no action.
    """
    import yaml
    import workshop_infrastructure.datasets.transformations as _transformations

    source = "<dict>"
    if not isinstance(info, dict):
        source = str(info)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Scalers file not found: {source}\n"
                "This is the data.scalers_path entry in your config. It is downloaded "
                "automatically on the first run; to fetch it explicitly:\n"
                "    python -m workshop_infrastructure.assets --scalers --dest <assets dir>"
            )
        with open(source, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)

    if not isinstance(info, dict):
        raise ValueError(f"Scalers source {source} must contain a mapping of channel name -> parameters.")

    available = sorted(
        name for name in dir(_transformations)
        if isinstance(getattr(_transformations, name), type)
    )

    ret_dict = {}
    for p_key, p_val in info.items():
        if not isinstance(p_val, dict) or "class" not in p_val:
            raise ValueError(
                f"Scalers entry {p_key!r} in {source} is malformed: expected a mapping "
                f"with a 'class' key, got {type(p_val).__name__}."
            )
        class_name = p_val["class"]
        # Note: p_val["base"] is intentionally not consulted — see the docstring.
        if not hasattr(_transformations, class_name):
            raise ValueError(
                f"Scalers entry {p_key!r} in {source} names class {class_name!r}, which does "
                "not exist in workshop_infrastructure.datasets.transformations.\n"
                f"Available classes: {', '.join(available)}."
            )
        ret_dict[p_key] = getattr(_transformations, class_name).from_dict(p_val)
    return ret_dict


HEAD_PREFIX = "head_"

# Normalization layers kept in fp32 by cast_frozen_to(). Both for numerical stability (the
# usual mixed-precision rule) and because F.layer_norm on CPU refuses a half weight with an
# fp32 input, which would break any pre-GPU forward pass in a notebook.
_NORM_LAYERS = (
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.LocalResponseNorm,
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.modules.instancenorm._InstanceNorm,
)


def discover_head_modules(model: torch.nn.Module) -> list[str]:
    """Find the fine-tuning head modules to keep trainable under LoRA.

    Convention: every trainable component of a fine-tuning head is a **direct
    child** of the top-level model whose attribute name starts with ``head_``
    (e.g. ``head_linear``, ``head_unembed``, ``head_cls_token``).  The backbone
    stays at ``backbone``.  Learners therefore never hand-maintain a list of
    head layers -- adding ``self.head_foo = nn.Linear(...)`` is enough.

    Only modules that own parameters are returned, so parameter-free layers
    such as ``head_dropout`` are not needlessly duplicated by PEFT.

    Raises:
        ValueError: if the model violates the convention, with a message
            naming the exact attribute to rename.
    """
    head_names = []
    missing_prefix = []
    for name, module in model.named_children():
        has_params = any(True for _ in module.parameters())
        if name.startswith(HEAD_PREFIX):
            if has_params:
                head_names.append(name)
        elif name != "backbone" and has_params:
            missing_prefix.append(name)

    if missing_prefix:
        raise ValueError(
            "Fine-tuning head modules must be named with the "
            f"{HEAD_PREFIX!r} prefix so LoRA can keep them trainable.\n"
            f"Rename these top-level attributes: "
            + ", ".join(f"self.{n} -> self.{HEAD_PREFIX}{n}" for n in missing_prefix)
            + "\nWithout the prefix they are frozen during LoRA fine-tuning and "
            "the model trains against a random readout."
        )

    # PEFT's modules_to_save cannot cover a bare nn.Parameter on the top-level
    # model -- it would stay frozen.  Wrap it in a tiny module instead.
    bare = [n for n, _ in model.named_parameters(recurse=False)]
    if bare:
        raise ValueError(
            "Top-level bare parameters cannot be kept trainable by PEFT, which "
            "matches module names only, so these would be silently frozen: "
            + ", ".join(bare)
            + "\nWrap each one in a small nn.Module (see ClassToken in "
            "workshop_infrastructure/models/finetune_models.py) and name the "
            f"attribute with the {HEAD_PREFIX!r} prefix."
        )

    # PEFT matches modules_to_save entries with a bare ``key.endswith(name)``
    # -- no dot boundary -- so a head name that happens to be a suffix of any
    # backbone module name would wrap that backbone module too.
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        collisions = [
            (head, f"backbone.{qualified}")
            for head in head_names
            for qualified, _ in backbone.named_modules()
            if qualified and f"backbone.{qualified}".endswith(head)
        ]
        if collisions:
            raise ValueError(
                "Head module names must not be a suffix of any backbone module "
                "name, because PEFT would make the backbone module trainable "
                "too:\n"
                + "\n".join(f"  {h} collides with {q}" for h, q in collisions)
                + "\nRename the head attribute to something more specific."
            )

    return head_names


def apply_peft_lora(
    model: torch.nn.Module,
    lora_config: LoraAdapterConfig,
) -> torch.nn.Module:
    """
    Applies PEFT LoRA adapters to a model.

    Adapters go on the modules named by ``lora_config.target_modules``.  Every
    fine-tuning head module (see :func:`discover_head_modules`) is passed to
    PEFT as ``modules_to_save``, so it stays **trainable** -- without this the
    head is frozen at its random initialisation and LoRA fits adapters to a
    random readout.

    Args:
        model: The model to apply LoRA to.
        lora_config: A LoraAdapterConfig instance (from workshop_infrastructure.configs).

    Returns:
        Model with PEFT LoRA adapters applied.
    """
    modules_to_save = discover_head_modules(model)

    print(
        f"Applying PEFT LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, modules={lora_config.target_modules}"
    )

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, peft_config)

    # Show exactly what is being trained, so a misconfigured run is visible
    # in the log instead of only in the loss curve.
    adapted = sorted(
        {
            name.split(".lora_A")[0].replace("base_model.model.", "")
            for name, _ in model.named_parameters()
            if ".lora_A" in name
        }
    )
    print(f"[LoRA] Adapted modules ({len(adapted)}):")
    for name in adapted:
        print(f"[LoRA]   {name}")
    print(f"[LoRA] Trainable head modules (modules_to_save): {modules_to_save}")

    # Defensive: current PEFT excludes modules_to_save from adapter injection.
    # If that ever changes, a head module would get both, so fail loudly.
    head_adapted = [n for n in adapted if n.split(".")[0].startswith(HEAD_PREFIX)]
    if head_adapted:
        raise RuntimeError(
            "PEFT applied LoRA adapters to fine-tuning head modules, which "
            f"should be fully trainable instead: {head_adapted}. "
            "Narrow lora_config.target_modules so it cannot match head layers."
        )

    # Log the number of trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    return model


def disable_peft_input_dtype_cast(model: torch.nn.Module) -> int:
    """Stop PEFT from upcasting activations to the adapter weight dtype.

    ``peft.tuners.lora.Linear.forward`` calls ``_cast_input_dtype(x, lora_A.weight.dtype)``
    before running the adapter branch.  Under mixed precision the adapter weights are
    fp32 while ``x`` is fp16/bf16, so PEFT allocates a fresh fp32 copy of the
    activation -- and autocast then casts it straight back down for ``lora_A``.  The
    copy buys nothing, and it is charged at the widest point in the network: for
    HelioSpectformer2D at 4096x4096, ``mlp.fc2``'s input is (1, 65536, 5120), so the
    copy is exactly 1.25 GiB, per adapted block, on every gradient-checkpoint recompute.

    ``peft.helpers.disable_input_dtype_casting`` does the same thing as a context
    manager, which cannot wrap ``trainer.fit()`` cleanly.  This sets the flag once.

    Only meaningful under autocast (``training.precision`` ending in ``-mixed``).  Under
    ``*-true`` precision the adapter weights are already half, ``x.dtype == dtype``, and
    PEFT skips the copy on its own -- so this becomes redundant rather than wrong.

    Args:
        model: A PEFT-wrapped model.  A model without adapters is a no-op.

    Returns:
        The number of tuner layers touched, so a silently ineffective call is visible
        in the log rather than only in the peak-memory number.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    touched = 0
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.cast_input_dtype_enabled = False
            touched += 1

    print(f"[LoRA] Disabled PEFT input-dtype casting on {touched} tuner layer(s).")
    return touched


def cast_frozen_to(model: torch.nn.Module, dtype: torch.dtype) -> tuple[int, int]:
    """Cast every frozen parameter and every float buffer to ``dtype``.

    Trainable parameters are deliberately left in fp32 so the optimizer keeps a
    full-precision master copy.  Under autocast the two dtypes coexist: autocast casts
    the fp32 adapter weights down for each matmul, and the frozen half-precision
    weights are already at the compute dtype.

    Beyond halving the frozen weights, this fixes two specific things on
    HelioSpectformer2D at 4096x4096:

      * ``LinearEmbedding.pos_embed`` is a (1, 65536, 1280) *buffer*, not a parameter --
        320 MiB in fp32.  ``embedding.py`` adds it to the half-precision patch-embedding
        output, and ``half + fp32 -> fp32``, which pins the entire inter-block residual
        stream to fp32 for every block.  With all 10 layers gradient-checkpointed that is
        11 live (1, 65536, 1280) tensors: 3.44 GiB in fp32 versus 1.72 GiB in half.
      * ``complex_weight`` (169 M elements, 645 MiB fp32) halves.  It is upcast back to
        fp32 inside ``SpectralGatingNetwork.forward`` for the FFT, so the rounding is
        permanent: measured relative error is 0.018% mean in fp16, 0.141% in bf16.  That
        is below the activation noise floor of mixed precision, but it *is* a change to a
        pretrained tensor, hence the print.

    Call this *after* :func:`load_pretrained_weights` and :func:`apply_peft_lora`, so the
    checkpoint is read at full precision and rounded exactly once, and so
    ``requires_grad`` already reflects the chosen fine-tuning regime.  Under full
    fine-tuning (``use_lora: false, freeze_backbone: false``) nothing is frozen and this
    only casts buffers.

    Normalization layers are deliberately left in fp32.  That is the standard
    mixed-precision rule -- their statistics are the numerically delicate part and they hold
    only ``2 * embed_dim`` parameters each, so excluding them costs well under a megabyte.
    It is also a hard requirement on CPU: ``F.layer_norm`` raises "mixed dtype (CPU):
    expect parameter to have scalar type of Float" when the weight is half and the input is
    fp32, which is exactly what a notebook's pre-GPU smoke test does.  (CUDA happens to
    tolerate it and returns fp32.)

    Note that after this call a bare ``model.forward(...)`` outside an autocast context
    raises ``expected scalar type Half but found Float`` from the patch-embedding Conv2d.
    Wrap ad-hoc forward passes in ``torch.autocast``.

    Args:
        model: The model to cast in place.
        dtype: ``torch.float16`` on pre-Ampere GPUs (SM75 emulates bf16),
            ``torch.bfloat16`` on SM80+.

    Returns:
        ``(n_params_cast, n_buffers_cast)``.
    """
    n_params = 0
    n_buffers = 0
    n_norm_skipped = 0

    # Walk modules rather than parameters() so normalization layers can be identified and
    # skipped wholesale, including their buffers (a BatchNorm's running statistics).
    for module in model.modules():
        if isinstance(module, _NORM_LAYERS):
            n_norm_skipped += 1
            continue
        # recurse=False: each module is visited once by the outer loop, so this assigns
        # every parameter to exactly one module and never re-casts through a parent.
        for param in module.parameters(recurse=False):
            if not param.requires_grad and param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(dtype)
                n_params += 1
        for buf in module.buffers(recurse=False):
            if buf is not None and buf.is_floating_point() and buf.dtype != dtype:
                buf.data = buf.data.to(dtype)
                n_buffers += 1

    print(
        f"[PRECISION] Cast {n_params} frozen parameter(s) and {n_buffers} buffer(s) to "
        f"{dtype}; left {n_norm_skipped} normalization layer(s) in fp32."
    )
    return n_params, n_buffers


def load_pretrained_weights(model: torch.nn.Module, pretrained_path: Optional[str]) -> None:
    """Load pretrained weights into a fine-tuning model, skipping shape-mismatched keys.

    The pretrained checkpoint was saved from HelioSpectFormer directly, so its keys
    are flat (e.g. ``embedding.proj.weight``).  The composed fine-tuning models
    (HelioSpectformer1D, HelioSpectformer2D) nest the backbone under ``backbone.*``,
    so we try both the original key and the ``backbone.``-prefixed key when matching
    against the current model's state dict.

    Args:
        model: The fine-tuning model to load weights into.
        pretrained_path: Path to the pretrained checkpoint (.pt file). No-op if None.
    """
    if not pretrained_path:
        return
    print(f"Loading pretrained weights from {pretrained_path}.")
    model_state = model.state_dict()
    checkpoint_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")

    remapped = {}
    for k, v in checkpoint_state.items():
        for candidate in (k, f"backbone.{k}"):
            if candidate in model_state and hasattr(v, "shape") and v.shape == model_state[candidate].shape:
                remapped[candidate] = v
                break

    model_state.update(remapped)
    model.load_state_dict(model_state, strict=True)
    print(f"Loaded {len(remapped)} / {len(checkpoint_state)} pretrained weights.")


class UploadBestCheckpointToS3(L.Callback):
    """Lightning callback that uploads the best checkpoint to S3 after each validation epoch.

    - No-op unless ``bucket`` is set (mirrors the ``output.s3_bucket: null`` YAML default).
    - Only runs from global rank 0 under DDP to avoid duplicate uploads.
    - Uses a stable S3 key (``fixed_key_name``) so the latest best is always at a
      predictable location regardless of epoch number.

    Args:
        checkpoint_cb: The ModelCheckpoint callback whose ``best_model_path`` to watch.
        bucket: S3 bucket name. Pass ``None`` to disable uploads entirely.
        prefix: Key prefix (folder) within the bucket, e.g. ``"flare/exp_001"``.
        fixed_key_name: Object name within the prefix. Defaults to ``"best.ckpt"``.
    """

    def __init__(
        self,
        checkpoint_cb: ModelCheckpoint,
        bucket: Optional[str],
        prefix: str = "",
        fixed_key_name: Optional[str] = "best.ckpt",
    ):
        super().__init__()
        self.checkpoint_cb = checkpoint_cb
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.fixed_key_name = fixed_key_name
        self.last_uploaded_best = None
        self._s3 = None

    def _upload_if_new_best(self, trainer) -> None:
        if not self.bucket:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        best_path = getattr(self.checkpoint_cb, "best_model_path", None)
        if not best_path or best_path == self.last_uploaded_best:
            return

        ckpt_path = Path(best_path)
        if not ckpt_path.exists():
            return

        if self._s3 is None:
            try:
                import boto3
            except ImportError as e:
                raise RuntimeError(
                    "boto3 is required for S3 uploads. Install with: pip install boto3"
                ) from e
            self._s3 = boto3.client("s3")

        object_name = self.fixed_key_name or ckpt_path.name
        s3_key = f"{self.prefix}/{object_name}" if self.prefix else object_name

        print(f"[S3] Uploading {ckpt_path} -> s3://{self.bucket}/{s3_key}")
        self._s3.upload_file(str(ckpt_path), self.bucket, s3_key)
        print("[S3] Upload complete.")
        self.last_uploaded_best = str(ckpt_path)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._upload_if_new_best(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Final check in case the last best update happens near fit end.
        self._upload_if_new_best(trainer)