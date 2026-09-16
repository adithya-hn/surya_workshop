import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append(".")

import torch
from workshop_infrastructure.utils import build_scalers

def main():
    torch.set_float32_matmul_precision('medium')

    from downstream_apps.sf_triggers.configs import load_flare_config
    cfg = load_flare_config("downstream_apps/sf_triggers/configs/config_script.yaml")
    print("config ok:", cfg.job_id)

    from workshop_infrastructure.assets import ensure_assets
    ensure_assets(cfg, which=["scalers"])
    scalers = build_scalers(info=cfg.data.scalers_path)
    print("scalers ok:", len(scalers))

    from downstream_apps.sf_triggers.datasets.sf_triggers_dataset import FlareDSDataset
    from workshop_infrastructure.datasets.builders import build_helio_dataloaders

    train_data_loader, val_data_loader = build_helio_dataloaders(
        cfg,
        FlareDSDataset,
        scalers=scalers,
        num_workers=4,
        return_surya_stack=True,
        max_number_of_samples=6,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
        ds_val_fraction=cfg.data.ds_val_fraction,
        ds_split_seed=cfg.data.ds_split_seed,
        mask_dir=cfg.data.mask_dir,
        mask_time_tolerance=cfg.data.mask_time_tolerance,
    )
    print("dataloaders ok")

    batch = next(iter(train_data_loader))
    print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in batch.items()})

if __name__ == "__main__":
    main()
